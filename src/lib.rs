#![no_std]
use core::{
    cell::UnsafeCell,
    mem::ManuallyDrop,
    ops::Range,
    sync::atomic::{AtomicI32, AtomicU32, Ordering},
};

extern crate alloc;
use alloc::{sync::Arc, vec::Vec};

use crossbeam_utils::CachePadded;

/// This is NOT a Vec<T>, but it's represnted as one for conveinence.
///
/// The `ptr` and `capacity` of the `Vec` are logically correct HOWEVER
/// the `len` field is not, instead the len field represents the length of
/// elements that will NOT be popped from the publication. Hence once
/// all elements have been popped, the `len` field will yet again represent
/// a valid `Vec` length.
#[repr(transparent)]
struct Publication<T>(ManuallyDrop<Vec<T>>);

impl<T> Default for Publication<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Publication<T> {
    unsafe fn read(&self, index: usize) -> T {
        unsafe { self.0.as_ptr().add(index).read() }
    }

    fn pop_offset(&self) -> usize {
        self.0.len()
    }
}

struct SharedState<T> {
    publication: UnsafeCell<Publication<T>>,
    head: CachePadded<AtomicI32>,
    tail: CachePadded<AtomicU32>,
}

/// Safety: The structure ensures that mutable, and immutable access of publication
/// don't happen at the same time.
unsafe impl<T: Send> Sync for SharedState<T> {}

impl<T> Default for SharedState<T> {
    fn default() -> Self {
        Self {
            publication: Default::default(),
            head: Default::default(),
            tail: Default::default(),
        }
    }
}

impl<T> SharedState<T> {
    /// # Safety
    /// - You must be the exclusive caller of this function.
    unsafe fn try_publish(&self, data: &mut Vec<T>) {
        // `Ordering::Acquire` ensures any reads to `publication` happen before any prospective writes.
        if self.tail.load(Ordering::Acquire) == 0 {
            // Safety: `SharedState::pop` only reads publication if it knows `tail != 0`.
            let publication = unsafe { self.publication.get().as_mut_unchecked() };

            let len = data.len();
            // This ratio is chosen arbitrarily.
            let pop_to = len / 2;
            // Safety: This is a little weird. In a publication the `len` field is actually
            // the final index we're allowed to pop. The _actual_ length is a combination of `self.tail`
            // and `self.head`. Once all (allowed) items have been popped the length of the vector will be
            // equal to the `pop_to` value: hence why we are safely useing the old publications vector as
            // our new local vector.
            // The reason we're doing this is because we want to share items between the local `Vec` and
            // the `Publication`, if we moved ALL the items then any future calls to pop from the local
            // `Vec` would fail (the local vec is our fast path).
            unsafe {
                data.set_len(pop_to);
            }

            core::mem::swap(data, &mut publication.0);

            // the number of elements that will be popped.
            let eff_len = (len - pop_to) as u32;
            // `Ordering::Relaxed` because `head` fences the publication and we don't need
            // to fence with the `tail.load` because this design only allows for a single producer.
            // This store needs to happen before head as readers always expect tail to be >= head.
            self.tail.store(eff_len, Ordering::Relaxed);
            // `Ordering::Release` ensures that readers can see the published data.
            self.head.store(eff_len as i32, Ordering::Release);
        }
    }

    fn pop(&self) -> Option<T> {
        // `Ordering::Acquire` ensures we see any writes to `publication`.
        let index = self.head.fetch_sub(1, Ordering::Acquire).wrapping_sub(1);
        if index < 0 {
            debug_assert_ne!(index, i32::MIN);
            return None;
        }

        // We must only fetch publication after we've checked that `head <= 0` because
        // the producer is allowed to publish if the tail is zero (and all we know is that the
        // tail is >= head).
        let publication = unsafe { self.publication.get().as_ref_unchecked() };
        let index = index as usize + publication.pop_offset();
        // Safety: `index` is guaranteed to be in bounds b/c `head` starts at the initalized length
        // and is decriment only and is checked against 0.
        let value = unsafe { publication.read(index) };
        // `Ordering::Release` ensures that this read happens before the decrement of `tail`.
        self.tail.fetch_sub(1, Ordering::Release);
        Some(value)
    }

    fn pop_n(&self, n: u32) -> PopN<'_, T> {
        // `Ordering::Acquire` ensures we see any writes to `publication`.
        let range_to = self.head.fetch_sub(n as i32, Ordering::Acquire);
        if range_to <= 0 {
            debug_assert!(range_to.wrapping_sub_unsigned(n) < 0);
            return PopN(None);
        }
        let range_to = range_to as u32;
        let range_from = range_to.saturating_sub(n);
        let n = range_from - range_to;

        // Safety: Since `tail >= head > 0` we know nobody is concurrently modifying the publication.
        let publication = unsafe { self.publication.get().as_ref_unchecked() };

        let range = (range_from as usize + publication.pop_offset())
            ..(range_to as usize + publication.pop_offset());
        // Safety: The `range` is guaranteed to be in bounds of the publication as it is bounded between
        // the initially set `head` value and initally set `pop_offset` value. And each index only occurs once.
        PopN(Some(unsafe {
            InnerPopN::new(publication, range, &self.tail, n)
        }))
    }
}

impl<T> Drop for SharedState<T> {
    fn drop(&mut self) {
        let publication = self.publication.get_mut();
        let len = *self.tail.get_mut() as usize + publication.pop_offset();

        for index in 0..len {
            // Safety: The number of elements in the publication is the addition
            // of the pop_offset and the `tail` count, so `index` is guaranteed to be in bounds.
            unsafe { drop(publication.read(index)) };
        }
    }
}

pub struct PopN<'a, T>(Option<InnerPopN<'a, T>>);

impl<'a, T> Iterator for PopN<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.as_mut().and_then(InnerPopN::next)
    }
}

struct InnerPopN<'a, T> {
    publication: &'a Publication<T>,
    range: Range<usize>,
    tail: &'a AtomicU32,
    n: u32,
}

impl<'a, T> InnerPopN<'a, T> {
    // Safety: The `range` must be in bounds of the publication.
    unsafe fn new(
        publication: &'a Publication<T>,
        range: Range<usize>,
        tail: &'a AtomicU32,
        n: u32,
    ) -> Self {
        Self {
            publication,
            range,
            tail,
            n,
        }
    }
}

impl<'a, T> Iterator for InnerPopN<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|index| {
            // Safety: Ensured by caller of `PopN::new`.
            unsafe { self.publication.read(index) }
        })
    }
}

impl<'a, T> Drop for InnerPopN<'a, T> {
    fn drop(&mut self) {
        for _ in self.into_iter() {}
        // `Ordering::Release` ensures that all reads happen before the decrement of `tail`.
        self.tail.fetch_sub(self.n, Ordering::Release);
    }
}

/// A reader for a [`FreeList`] that provides methods for popping values.
pub struct FreeListReader<T> {
    shared: Arc<SharedState<T>>,
}

impl<T> FreeListReader<T> {
    /// Pop a single value from the free list.
    pub fn pop(&self) -> Option<T> {
        self.shared.pop()
    }

    /// Pop up to `n` values from the free list.
    pub fn pop_n(&self, n: u32) -> PopN<'_, T> {
        self.shared.pop_n(n)
    }
}

pub struct FreeList<T> {
    shared: Arc<SharedState<T>>,
    local: Vec<T>,
}

impl<T> core::ops::Deref for FreeList<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.local
    }
}

impl<T> core::ops::DerefMut for FreeList<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.local
    }
}

impl<T> Default for FreeList<T> {
    fn default() -> Self {
        Self {
            shared: Default::default(),
            local: Default::default(),
        }
    }
}

impl<T> FreeList<T> {
    /// Creates a new `FreeList`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a `FreeListReader` that can be used to pop from the `FreeList`.
    pub fn reader(&self) -> FreeListReader<T> {
        FreeListReader {
            shared: self.shared.clone(),
        }
    }

    /// Synchronizes the local `FreeList` with the shared state.
    ///
    /// This method must be called frequently. How frequently depends on your usage.
    /// To be specific:
    /// 1. The remote free list will never have any items.
    /// 2. If this is not called between i32::MAX failed remote pop calls there will be undefined behavior.
    pub fn sync(&mut self) {
        // Safety:
        unsafe {
            self.shared.try_publish(&mut self.local);
        }
    }

    // I do not provide methods for remote popping as you should always pop locally. If you need
    // strong guarantees (that popping will always return a value if there is one) then you should
    // not use this type.
}
