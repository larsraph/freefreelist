#![no_std]
use core::{
    cell::UnsafeCell,
    iter::FusedIterator,
    mem::{self, ManuallyDrop},
    ops::{Deref, DerefMut, Range},
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
#[derive(Debug)]
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

#[derive(Debug)]
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

            mem::swap(data, &mut publication.0);

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
            debug_assert_ne!(index, i32::MIN, "head overflow");
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
        PopN::new(self, n)
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

#[derive(Debug)]
pub struct PopN<'a, T> {
    publication: &'a UnsafeCell<Publication<T>>,
    tail: &'a AtomicU32,
    range: Range<i32>,
    poped: u32,
}

impl<'a, T> PopN<'a, T> {
    fn new(shared: &'a SharedState<T>, n: u32) -> Self {
        // `Ordering::Acquire` ensures we see any writes to `publication`.
        let range_to = shared.head.fetch_sub(n as i32, Ordering::Acquire);
        let range_from = range_to.wrapping_sub_unsigned(n);
        debug_assert!(range_from < range_to, "head overflow");
        // When less than zero bring it back to zero... unless range_to is also less than zero,
        // in which case they should be equal (so that poped is zero and range.next() returns None).
        let range_from = range_from.max(0).min(range_to);
        let poped = (range_to - range_from) as u32;
        let range = range_from..range_to;

        Self {
            publication: &shared.publication,
            tail: &shared.tail,
            range,
            poped,
        }
    }
}

impl<'a, T> Iterator for PopN<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|index| {
            // Safety: During creation of `PopN` our `range` only has values if the `head > 0`.
            // We do this now and not in `PopN::new` because in `PopN::new` we don't want to
            // branch on `poped > 0`.
            let publication = unsafe { self.publication.get().as_ref_unchecked() };
            // usize cast doesn't wrap because if `range_to` is less than zero then `range_from == range_to` and no items will be yielded.
            let index = index as usize + publication.pop_offset();
            // Safety: `index` is within range which is bounded to `head + pop_offset` and `pop_offset` and no index is
            // poped twice.
            unsafe { publication.read(index) }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for PopN<'a, T> {}

impl<'a, T> FusedIterator for PopN<'a, T> {}

impl<'a, T> Drop for PopN<'a, T> {
    fn drop(&mut self) {
        for _ in self.into_iter() {}
        // `Ordering::Release` ensures that all reads happen before the decrement of `tail`.
        self.tail.fetch_sub(self.poped, Ordering::Release);
    }
}

/// A reader for a [`FreeList`] that provides methods for popping values.
#[derive(Debug, Clone)]
pub struct FreeListReader<T> {
    shared: Arc<SharedState<T>>,
}

impl<T> PartialEq for FreeListReader<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }
}

impl<T> Eq for FreeListReader<T> {}

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

#[derive(Debug)]
pub struct FreeList<T> {
    shared: Arc<SharedState<T>>,
    local: Vec<T>,
}

impl<T> Deref for FreeList<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.local
    }
}

impl<T> DerefMut for FreeList<T> {
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
    ///
    /// It checks if the `SharedState` is drained. If so it swaps the local `Vec` with the shared `Vec`.
    pub fn sync(&mut self) {
        // Safety: We have exclusive access to `self` and this type is the only
        // type that can publicly call this funciton.
        unsafe {
            self.shared.try_publish(&mut self.local);
        }
    }
}
