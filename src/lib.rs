#![no_std]
use core::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicI32, AtomicU32, Ordering},
};

extern crate alloc;

use alloc::{boxed::Box, vec::Vec};

use crossbeam_utils::CachePadded;

/// A buffer allocation of T
struct Publication<T> {
    data: Box<[MaybeUninit<T>]>,
}

impl<T> Default for Publication<T> {
    fn default() -> Self {
        Self { data: Box::new([]) }
    }
}

impl<T> Publication<T> {
    fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Writes data to the publication. This will grow (amortized) the publication to fit the data.
    ///
    /// # Safety
    /// - caller must ensure that `len == 0`
    unsafe fn publish_assume_empty(&mut self, data: impl ExactSizeIterator<Item = T>) {
        let trgt_cap = data.len();
        let curr_cap = self.capacity();
        if trgt_cap > curr_cap {
            let amortized_cap = curr_cap * 2;
            let new_cap = amortized_cap.max(trgt_cap);

            self.data = (0..new_cap).map(|_| MaybeUninit::uninit()).collect();
        }

        for (index, val) in data.enumerate() {
            // Safety: We just ensured that cap > len
            unsafe {
                self.data.get_unchecked_mut(index).write(val);
            }
        }
    }

    /// # Safety
    /// - caller must ensure that `index < len` (and by extension, `index < self.capacity()`)
    unsafe fn get_unchecked(&self, index: usize) -> T {
        // Safety: Ensured by caller
        unsafe { self.data.get_unchecked(index).as_ptr().read() }
    }
}

pub struct RemoteFreeList<T> {
    publication: UnsafeCell<Publication<T>>,
    claim: CachePadded<AtomicI32>,
    len: CachePadded<AtomicU32>,
}

unsafe impl<T: Send> Sync for RemoteFreeList<T> {}

impl<T> Default for RemoteFreeList<T> {
    fn default() -> Self {
        Self {
            publication: UnsafeCell::new(Publication::default()),
            claim: CachePadded::new(AtomicI32::new(0)),
            len: CachePadded::new(AtomicU32::new(0)),
        }
    }
}

impl<T> RemoteFreeList<T> {
    /// Returns weather or not all values of the Publication have been poped.
    pub fn is_empty(&self) -> bool {
        self.len.load(Ordering::Acquire) == 0
    }

    /// Returns the next value from the list. Keep in mind that this is a best effort pop and
    /// may return `None` even if there are recyclable values.
    ///
    /// If you call this function a large number of times (`i32::MAX - 1`) before a single `sync`
    /// call in the owning `FreeList` the behavior is undefined.
    pub fn pop(&self) -> Option<T> {
        let index = self.claim.fetch_sub(1, Ordering::Acquire).wrapping_sub(1);
        if index < 0 {
            // This should basically never happen BUT is completely dependent on the usage patterns of the caller.
            debug_assert_ne!(
                index,
                i32::MIN,
                "`claim` overflow; more than {} pops have occurred since the last len check",
                i32::MIN.abs()
            );
            return None;
        }

        // Safety: The publication is only modified when all items have been (completely) popped and we are
        // in the middle of popping a value. And the index is obtained from a decriment only value that
        // was initalized to the `len` and so is ensured to be in bounds.
        let value = unsafe {
            self.publication
                .get()
                .as_ref_unchecked()
                .get_unchecked(index as usize)
        };

        // Cannot overflow because len is always >= claim and we don't decrement unless claim >= 0.
        // We must decriment this separately than `claim` because we can only write a new
        // publication when all items have been claimed AND read.
        self.len.fetch_sub(1, Ordering::Release);

        Some(value)
    }

    /// # Safety
    /// - caller must ensure that `len == 0`
    /// - must not be called concurrently with itself (exclusive publisher)
    unsafe fn publish_assume_exclusive_empty(&self, data: impl ExactSizeIterator<Item = T>) {
        let len = data.len();
        debug_assert!(len < i32::MAX as usize);
        // Safety: ensured by caller
        unsafe {
            self.publication
                .get()
                .as_mut_unchecked()
                .publish_assume_empty(data);
        }
        // len must be updated first to ensure that poppers don't see a stale `len` value but a
        // up-to-date `claim` value.
        // Is `Ordering::Relaxed` because `claim` is the atomic boundary.
        self.len.store(len as u32, Ordering::Relaxed);
        // `Ordering::Release` to ensure that the `publication` is visible to other threads.
        self.claim.store(len as i32, Ordering::Release);
    }
}

impl<T> Drop for RemoteFreeList<T> {
    fn drop(&mut self) {
        let len = *self.len.get_mut();
        let publication = self.publication.get_mut();
        for index in 0..len as usize {
            // Safety: len is the number of initalized items in the collection.
            let val = unsafe { publication.get_unchecked(index) };
            drop(val);
        }
    }
}

/// A `FreeList` that can hand out a `RemoteFreeList` to cheaply pop items from any thread.
pub struct FreeList<'a, T> {
    remote: &'a RemoteFreeList<T>,
    local: Vec<T>,
}

impl<T> FreeList<'static, T> {
    /// Leaks a `RemoteFreeList` and returns a `FreeList` that uses it.
    pub fn new_leaked() -> Self {
        let remote = Box::leak(Box::new(RemoteFreeList::default()));
        Self::new(remote)
    }
}

impl<'a, T> FreeList<'a, T> {
    /// Creates a new `FreeList` with a `RemoteFreeList` of lifetime `'a`.
    ///
    /// Takes a mutable reference to the `RemoteFreeList` to ensure only one `FreeList`
    /// receives ownership of the `RemoteFreeList`.
    pub fn new(remote: &'a mut RemoteFreeList<T>) -> Self {
        Self {
            remote,
            local: Vec::new(),
        }
    }

    /// Returns a reference to the `RemoteFreeList`.
    pub fn get_remote(&self) -> &'a RemoteFreeList<T> {
        self.remote
    }

    fn drain_half(&mut self) -> impl ExactSizeIterator<Item = T> {
        let range_from = self.local.len() / 2;
        self.local.drain(range_from..)
    }

    /// If the `RemoteFreeList` is empty then we drain half of the `local` FreeList into it.
    pub fn sync(&mut self) {
        if self.remote.is_empty() {
            // Safety: We just checked `is_empty()`.
            // Since we have `&mut self`, we know no other thread can call
            // `assume_exclusive_empty_publish` concurrently.
            unsafe {
                self.remote
                    .publish_assume_exclusive_empty(self.drain_half());
            };
        }
    }

    /// Pushes an item onto the `FreeList`.
    ///
    /// Does NOT affect the `RemoteFreeList`.
    pub fn push_local(&mut self, item: T) {
        self.local.push(item);
    }

    /// Pushes an item onto the `FreeList` and synchronizes with the `RemoteFreeList` every `N` items.
    pub fn push_sync_every<const N: usize>(&mut self, item: T) {
        self.push_local(item);
        if self.local.len().is_multiple_of(N) {
            self.sync();
        }
    }

    /// Pushes and item onto the `FreeList` and synchronizes with the `RemoteFreeList`.
    pub fn push_sync(&mut self, item: T) {
        self.push_sync_every::<1>(item);
    }

    /// Pops an item from the `local` `FreeList`. Keep in mind that this is a best-effort pop.
    /// This may return `None` even if the `FreeList` is has items.
    pub fn pop_local(&mut self) -> Option<T> {
        self.local.pop()
    }

    /// Pops an item from the `RemoteFreeList`. Keep in mind that this is a best-effort pop.
    /// This may return `None` even if the `RemoteFreeList` is has items.
    pub fn pop_remote(&self) -> Option<T> {
        self.remote.pop()
    }
}
