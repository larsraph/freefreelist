#![no_std]
use core::{
    cell::UnsafeCell,
    iter::FusedIterator,
    mem::{self, ManuallyDrop},
    ops::{Deref, DerefMut, Range},
    sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering},
};

extern crate alloc;
use alloc::{sync::Arc, vec::Vec};

use crossbeam_utils::CachePadded;

/// A `Vec` that is in the process of being drained.
#[repr(transparent)]
#[derive(Debug)]
struct Publication<T>(ManuallyDrop<Vec<T>>);

impl<T> Default for Publication<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Publication<T> {
    /// Swaps the previous `Vec` as if it had been drained to the previous `drain_from` index
    /// with the `other` `Vec` that will be drained to `drain_from`.
    ///
    /// # Safety
    /// - must have been drained to the previous `drain_from` value
    /// - `drain_from` must be less than or equal to the length of `other`
    unsafe fn swap_as_drained(&mut self, other: &mut Vec<T>, drain_from: usize) {
        // Safety: the next time `swap_as_drained` is called this length will be correct
        unsafe {
            other.set_len(drain_from);
        }
        mem::swap(&mut *self.0, other);
    }

    /// Returns the value at `index` in the `Vec`.
    ///
    /// # Safety
    /// - `index` must be less than the length of the `Vec`
    /// - the item at `index` must have been previously `initalized`
    /// - the item at `index` must not have been previously `read`
    unsafe fn read(&self, index: usize) -> T {
        unsafe { self.0.as_ptr().add(index).read() }
    }

    /// Returns the final length of the `Vec` after draining.
    fn drain_from(&self) -> usize {
        self.0.len()
    }
}

/// A wrapper around a `Vec` that facilitates mulitple threads popping
/// values concurrently as well as swapping in a new `Vec` when the old
/// one is drained.
#[derive(Debug)]
struct SharedPopVec<T> {
    /// The `Vec` being drained. As long as `tail != 0` this is immutable. When `tail == 0` the
    /// single producer has exclusive mutable access.
    publication: UnsafeCell<Publication<T>>,
    /// The number of items that are initalized and not currently being drained.
    /// Readers decriment this to reserve an item for reading.
    head: CachePadded<AtomicI32>,
    /// The number of items remaining to be drained from `publication`.
    /// The `tail` is always `>= head` because they are initalized to the same
    /// value and the `tail` is decrimented after the `head` is decrimented.
    tail: CachePadded<AtomicU32>,
}

impl<T> Default for SharedPopVec<T> {
    fn default() -> Self {
        Self {
            publication: Default::default(),
            head: Default::default(),
            tail: Default::default(),
        }
    }
}

impl<T> SharedPopVec<T> {
    /// This just panics.
    /// It is included to help with branch prediction, and put the panic message in one spot.
    #[cold]
    #[inline]
    fn on_overflow() -> ! {
        panic!("head overflow")
    }

    pub fn pop(&self) -> Option<T> {
        // `Ordering::Acquire` ensures we see any writes to `publication`.
        let index = self.head.fetch_sub(1, Ordering::Acquire).wrapping_sub(1);
        if index < 0 {
            if index == i32::MAX {
                Self::on_overflow();
            }
            return None;
        }

        // Safety: `publication` is under shared ownership because `tail >= head > 0`.
        let publication = unsafe { self.publication.get().as_ref_unchecked() };
        let index = index as usize + publication.drain_from();
        // Safety: `index` is reserved from `head` and so is guaranteed to be in range and unread.
        let value = unsafe { publication.read(index) };
        // `Ordering::Release` ensures that the `publication` read happens before the decrement of `tail`.
        self.tail.fetch_sub(1, Ordering::Release);
        Some(value)
    }

    fn pop_n(&self, n: u32) -> InnerPopN<'_, T> {
        InnerPopN::new(self, n)
    }

    /// # Safety
    /// - You must be the exclusive publisher to call this function.
    /// - The `tail` MUST be equal to zero to avoid race conditions.
    unsafe fn publish(&self, data: &mut Vec<T>) {
        // Safety: When tail is zero the single publisher gets exclusive access to the `publication`.
        let publication = unsafe { self.publication.get().as_mut_unchecked() };

        let len = data.len();
        let drain_from = data.len() / 2;

        // Safety:
        // - `drain_from` is guaranteed to be less or equal to than `data.len()`.
        // - caller ensures that the publication was fully drained before swapping.
        unsafe {
            publication.swap_as_drained(data, drain_from);
        }

        let eff_len = (len - drain_from) as u32;
        // `Ordering::Relaxed` because `head` fences the publication and we don't need
        // to fence with `tail.load`s because this design only allows for a single producer.
        // This store needs to happen before head as readers always expect tail to be >= head.
        self.tail.store(eff_len, Ordering::Relaxed);
        // `Ordering::Release` ensures that readers can see the published data.
        self.head.store(eff_len as i32, Ordering::Release);
    }
}

impl<T> Drop for SharedPopVec<T> {
    fn drop(&mut self) {
        let publication = self.publication.get_mut();
        let tail = self.tail.get_mut();
        let len = *tail as usize + publication.drain_from();

        for i in 0..len {
            // Safety: the length is the number of initalized and unpoped elements in the publication.
            unsafe {
                drop(publication.read(i));
            }
        }
    }
}

/// Safety: The structure ensures that mutable, and immutable access of publication
/// don't happen at the same time.
unsafe impl<T: Send> Sync for SharedPopVec<T> {}

#[derive(Debug)]
struct InnerPopN<'a, T> {
    shared: &'a SharedPopVec<T>,
    range: Range<i32>,
    popped: u32,
}

impl<'a, T> InnerPopN<'a, T> {
    fn new(shared: &'a SharedPopVec<T>, n: u32) -> Self {
        if n == 0 {
            return Self {
                shared,
                range: 0..0,
                popped: 0,
            };
        }

        // `Ordering::Acquire` ensures we see any writes to `publication`.
        let range_to = shared.head.fetch_sub(n as i32, Ordering::Acquire);
        let range_from = range_to.wrapping_sub_unsigned(n);
        if range_from >= range_to {
            SharedPopVec::<T>::on_overflow();
        }
        // This ensures that `range_from` is greater than `range_to` when `head` is less than zero,
        // so that `range.next()` returns `None`.
        let range_from = range_from.max(0);
        // This ensures that when `range_from` is forced to be greater than `range_to` (by `max(0)`),
        // `popped` is still zero.
        let popped = (range_to.wrapping_sub(range_from)).max(0) as u32;
        let range = range_from..range_to;

        Self {
            shared,
            range,
            popped,
        }
    }
}

impl<'a, T> Iterator for InnerPopN<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|index| {
            // Safety: During the creation of `PopN` our `range` only has values if `head > 0`.
            // publication is shared immutable when `tail > 0` and `tail >= head.
            let publication = unsafe { self.shared.publication.get().as_ref_unchecked() };
            // `index` doesn't wrap because `range_from` is bound to `0`.
            let index = index as usize + publication.drain_from();
            // Safety: the `range` is reserved from `head` and bound by `0`.
            unsafe { publication.read(index) }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for InnerPopN<'a, T> {}
impl<'a, T> FusedIterator for InnerPopN<'a, T> {}

impl<'a, T> Drop for InnerPopN<'a, T> {
    fn drop(&mut self) {
        for _ in self.into_iter() {}
        // `Ordering::Release` ensures that all reads happen before the decrement of `tail`.
        if self.popped != 0 {
            self.shared.tail.fetch_sub(self.popped, Ordering::Release);
        }
    }
}

#[derive(Debug)]
struct SharedState<T> {
    a: SharedPopVec<T>,
    b: SharedPopVec<T>,
    prioritize_b: CachePadded<AtomicBool>,
}

impl<T> Default for SharedState<T> {
    fn default() -> Self {
        Self {
            a: Default::default(),
            b: Default::default(),
            prioritize_b: Default::default(),
        }
    }
}

impl<T> SharedState<T> {
    fn pop(&self) -> Option<T> {
        if self.prioritize_b.load(Ordering::Relaxed) {
            self.b.pop().or_else(|| self.a.pop())
        } else {
            self.a.pop().or_else(|| self.b.pop())
        }
    }

    fn pop_n(&self, n: u32) -> PopN<'_, T> {
        if self.prioritize_b.load(Ordering::Relaxed) {
            let a = self.b.pop_n(n);
            let rem = n - a.len() as u32;
            let b = self.a.pop_n(rem);
            PopN { a, b }
        } else {
            let a = self.a.pop_n(n);
            let rem = n - a.len() as u32;
            let b = self.b.pop_n(rem);
            PopN { a, b }
        }
    }

    /// # Safety
    /// - You must be the exclusive publisher
    unsafe fn try_publish(&self, data: &mut Vec<T>) {
        let a = self.a.tail.load(Ordering::Relaxed) == 0;
        let b = self.b.tail.load(Ordering::Relaxed) == 0;
        if !a && !b {
            return;
        }

        let l = data.len() == 0;
        if a && b && l {
            return;
        }

        let prioritize_b = self.prioritize_b.load(Ordering::Relaxed);
        let buffers = if prioritize_b {
            [(&self.b, b), (&self.a, a)]
        } else {
            [(&self.a, a), (&self.b, b)]
        };
        if buffers[0].1 {
            self.prioritize_b.fetch_not(Ordering::Relaxed);
        }
        for (buffer, is_empty) in buffers {
            if is_empty {
                continue;
            }
            // `Acquire` any writes. We don't need the actual value because we already checked that the value
            // was zero and once the `tail` is 0 the only possible writer is the single producer (aka this function)
            let _ = buffer.tail.load(Ordering::Acquire);
            // Safety: Caller ensures they are the only publisher.
            // We checked if the prioritized buffer is empty.
            unsafe {
                buffer.publish(data);
            }
        }
    }
}

/// Manual Chain operation. Theoretically more optimizable because Chain uses Option<T> for portability.
pub struct PopN<'a, T> {
    a: InnerPopN<'a, T>,
    b: InnerPopN<'a, T>,
}

impl<'a, T> Iterator for PopN<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.a.next().or_else(|| self.b.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.a.size_hint().0 + self.b.size_hint().0;
        (len, Some(len))
    }
}

impl<'a, T> ExactSizeIterator for PopN<'a, T> {}
impl<'a, T> FusedIterator for PopN<'a, T> {}

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
