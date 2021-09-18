//! Atomic References
//!
//! These types act similarially to the Atomic types from std::sync::atomic,
//! Except that instead of containing an integer type or a pointer, they contain
//! an `Option<&'a T>` value.
//!
//! Like other option values, these types present operations which, when used
//! correctly, synchronize updates between threads. This type is a form of
//! interior mutability, like `Cell<T>`, `RefCell<T>`, or `Mutex<T>`.
//!
//! This type in static position is often used for lazy global initialization.
//!
//! `AtomicRef` may only contain `Sized` types, as unsized types have wide
//! pointers which cannot be atomically written to or read from.
//!
//! # Ordering
//!
//! It is unsound to load or store an atomic reference with the `Relaxed` memory
//! ordering, as these operations provide no ordering on writes to the data
//! behind the reference. To avoid this issue, loads and stores with `Relaxed`
//! memory ordering are actually performed with `Acquire`, `Release`, or
//! `AcqRel` ordering, as appropriate.
//!
//! # Examples
//!
//! Static logger state
//!
//! ```
//! use atomic_ref::AtomicRef;
//! use std::sync::atomic::Ordering;
//! use std::io::{stdout, Write};
//!
//! // Define the idea of a logger
//! trait Logger {
//!     fn log(&self, msg: &str) {}
//! }
//! struct LoggerInfo {
//!     logger: &'static (dyn Logger + Sync)
//! }
//!
//! // The methods for working with our currently defined static logger
//! static LOGGER: AtomicRef<LoggerInfo> = AtomicRef::new(None);
//! fn log(msg: &str) -> bool {
//!     if let Some(info) = LOGGER.load(Ordering::SeqCst) {
//!         info.logger.log(msg);
//!         true
//!     } else {
//!         false
//!     }
//! }
//! fn set_logger(logger: Option<&'static LoggerInfo>) {
//!     LOGGER.store(logger, Ordering::SeqCst);
//! }
//!
//! // Defining the standard out example logger
//! struct StdoutLogger;
//! impl Logger for StdoutLogger {
//!     fn log(&self, msg: &str) {
//!         stdout().write(msg.as_bytes());
//!     }
//! }
//! static STDOUT_LOGGER: LoggerInfo = LoggerInfo { logger: &StdoutLogger };
//!
//! fn main() {
//!     let res = log("This will fail");
//!     assert!(!res);
//!     set_logger(Some(&STDOUT_LOGGER));
//!     let res = log("This will succeed");
//!     assert!(res);
//! }
//! ```
#![no_std]

use core::default::Default;
use core::fmt;
use core::marker::PhantomData;
use core::mem;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

/// A mutable `Option<&'a T>` type which can be safely shared between threads.
#[repr(C)]
pub struct AtomicRef<'a, T: 'a> {
    data: AtomicPtr<T>,
    // Make `AtomicRef` invariant over `'a` and `T`
    _marker: PhantomData<Invariant<'a, T>>,
}

// Work-around for the construction of `PhantomData<&mut _>` requiring
// `#![feature(const_fn)]`
struct Invariant<'a, T: 'a>(&'a mut &'a T);

/// An internal helper function for converting `Option<&'a T>` values to
/// `*mut T` for storing in the `AtomicUsize`.
#[inline(always)]
const fn from_opt<'a, T>(p: Option<&'a T>) -> *mut T {
    match p {
        Some(p) => p as *const T as *mut T,
        None => ptr::null_mut(),
    }
}

/// An internal helper function for converting `*mut T` values stored in the
/// `AtomicUsize` back into `Option<&'a T>` values.
#[inline(always)]
unsafe fn to_opt<'a, T>(p: *mut T) -> Option<&'a T> {
    p.as_ref()
}

// As noted in #5, the use of `Relaxed` ordering with `atomic_ref` is unsound,
// as `Relaxed` ordering performs no synchronization on the data behind the
// reference. These methods restrict the load ordering to safe orderings by
// requiring at least `Acquire` ordering for load operations, and `Release`
// ordering for store operations.

/// Restrict memory ordering for atomic load operations.
#[inline]
fn enforce_load_ordering(order: Ordering) -> Ordering {
    match order {
        Ordering::Relaxed | Ordering::Acquire => Ordering::Acquire,
        Ordering::SeqCst => Ordering::SeqCst,
        Ordering::Release => panic!("there is no such thing as a release load"),
        Ordering::AcqRel => panic!("there is no such thing as an acquire/release load"),
        _ => panic!("unsupported memory ordering: {:?}", order),
    }
}

/// Restrict memory ordering for atomic store operations.
#[inline]
fn enforce_store_ordering(order: Ordering) -> Ordering {
    match order {
        Ordering::Relaxed | Ordering::Release => Ordering::Release,
        Ordering::SeqCst => Ordering::SeqCst,
        Ordering::Acquire => panic!("there is no such thing as an acquire store"),
        Ordering::AcqRel => panic!("there is no such thing as an acquire/release store"),
        _ => panic!("unsupported memory ordering: {:?}", order),
    }
}

/// Restrict memory ordering for atomic RMW operations.
#[inline]
fn enforce_swap_ordering(order: Ordering) -> Ordering {
    match order {
        Ordering::Relaxed | Ordering::Acquire | Ordering::Release | Ordering::AcqRel => {
            Ordering::AcqRel
        }
        Ordering::SeqCst => Ordering::SeqCst,
        _ => panic!("unsupported memory ordering: {:?}", order),
    }
}

impl<'a, T> AtomicRef<'a, T> {
    /// Creates a new `AtomicRef`.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomic_ref::AtomicRef;
    ///
    /// static VALUE: i32 = 10;
    /// let atomic_ref = AtomicRef::new(Some(&VALUE));
    /// ```
    pub const fn new(p: Option<&'a T>) -> AtomicRef<'a, T> {
        AtomicRef {
            data: AtomicPtr::new(from_opt(p)),
            _marker: PhantomData,
        }
    }

    /// Returns a mutable reference to the underlying `Option<&'a T>`.
    ///
    /// This is safe because the mutable reference guarantees that no other
    /// threads are concurrently accessing the atomic data.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::Ordering;
    /// use atomic_ref::AtomicRef;
    ///
    /// let value: i32 = 10;
    /// let value2: i32 = 20;
    ///
    /// let mut some_ref = AtomicRef::new(Some(&value));
    /// assert_eq!(*some_ref.get_mut(), Some(&value));
    /// *some_ref.get_mut() = Some(&value2);
    /// assert_eq!(some_ref.load(Ordering::SeqCst), Some(&value2));
    /// ```
    pub fn get_mut(&mut self) -> &mut Option<&'a T> {
        debug_assert_eq!(mem::size_of::<Option<&'a T>>(), mem::size_of::<*mut T>());
        unsafe { mem::transmute::<&mut *mut T, &mut Option<&'a T>>(self.data.get_mut()) }
    }

    /// Consumes the atomic and returns the contained value.
    ///
    /// This is safe because passing `self` by value guarantees that no other
    /// threads are concurrently accessing the atomic data.
    ///
    /// # Examples
    ///
    /// ```
    /// use atomic_ref::AtomicRef;
    ///
    /// let some_ref = AtomicRef::new(Some(&5));
    /// assert_eq!(some_ref.into_inner(), Some(&5));
    /// ```
    pub fn into_inner(self) -> Option<&'a T> {
        unsafe { to_opt(self.data.into_inner()) }
    }

    /// Loads the value stored in the `AtomicRef`.
    ///
    /// `load` takes an `Ordering` argument which describes the memory ordering
    /// of this operation.
    ///
    /// Calls with ordering weaker than `Acquire` will be performed with
    /// `Acquire` ordering.
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Release` or `AcqRel`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::Ordering;
    /// use atomic_ref::AtomicRef;
    ///
    /// static VALUE: i32 = 10;
    ///
    /// let some_ref = AtomicRef::new(Some(&VALUE));
    /// assert_eq!(some_ref.load(Ordering::Acquire), Some(&10));
    /// ```
    pub fn load(&self, ordering: Ordering) -> Option<&'a T> {
        unsafe { to_opt(self.data.load(enforce_load_ordering(ordering))) }
    }

    /// Stores a value into the `AtomicRef`.
    ///
    /// `store` takes an `Ordering` argument which describes the memory ordering
    /// of this operation.
    ///
    /// Calls with ordering weaker than `Release` will be performed with
    /// `Release` ordering.
    ///
    /// # Panics
    ///
    /// Panics if `order` is `Acquire` or `AcqRel`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::Ordering;
    /// use atomic_ref::AtomicRef;
    ///
    /// static VALUE: i32 = 10;
    ///
    /// let some_ptr = AtomicRef::new(None);
    /// some_ptr.store(Some(&VALUE), Ordering::SeqCst);
    /// ```
    pub fn store(&self, ptr: Option<&'a T>, order: Ordering) {
        self.data
            .store(from_opt(ptr), enforce_store_ordering(order))
    }

    /// Stores a value into the `AtomicRef`, returning the old value.
    ///
    /// `swap` takes an `Ordering` argument which describes the memory ordering
    /// of this operation.
    ///
    /// Calls with ordering weaker than `AcqRel` will be performed with `AcqRel`
    /// ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::Ordering;
    /// use atomic_ref::AtomicRef;
    ///
    /// static VALUE: i32 = 10;
    /// static OTHER_VALUE: i32 = 20;
    ///
    /// let some_ptr = AtomicRef::new(Some(&VALUE));
    /// let value = some_ptr.swap(Some(&OTHER_VALUE), Ordering::SeqCst);
    /// ```
    pub fn swap(&self, p: Option<&'a T>, order: Ordering) -> Option<&'a T> {
        unsafe { to_opt(self.data.swap(from_opt(p), enforce_swap_ordering(order))) }
    }

    /// Stores a value into the `AtomicRef` if the current value is the "same" as
    /// the `current` value.
    ///
    /// The return value is always the previous value. If it the "same" as
    /// `current`, then the value was updated.
    ///
    /// This method considers two `Option<&'a T>`s to be the "same" if they are
    /// both `Some` and have the same pointer value, or if they are both `None`.
    /// This method does not use `Eq` or `PartialEq` for comparison.
    ///
    /// `compare_and_swap` also takes an `Ordering` argument which describes the
    /// memory ordering of this operation.
    ///
    /// Calls with ordering weaker than `AcqRel` will be performed with `AcqRel`
    /// ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::Ordering;
    /// use atomic_ref::AtomicRef;
    ///
    /// static VALUE: i32 = 10;
    /// static OTHER_VALUE: i32 = 20;
    ///
    /// let some_ptr = AtomicRef::new(Some(&VALUE));
    /// let value = some_ptr.compare_and_swap(Some(&OTHER_VALUE), None, Ordering::SeqCst);
    /// ```
    #[allow(deprecated)]
    pub fn compare_and_swap(
        &self,
        current: Option<&'a T>,
        new: Option<&'a T>,
        order: Ordering,
    ) -> Option<&'a T> {
        unsafe {
            to_opt(self.data.compare_and_swap(
                from_opt(current),
                from_opt(new),
                enforce_swap_ordering(order),
            ))
        }
    }

    /// Stores a value into the `AtomicRef` if the current value is the "same" as
    /// the `current` value.
    ///
    /// The return value is a result indicating whether the new value was
    /// written, and containing the previous value. On success this value is
    /// guaranteed to be the "same" as `new`.
    ///
    /// This method considers two `Option<&'a T>`s to be the "same" if they are
    /// both `Some` and have the same pointer value, or if they are both `None`.
    /// This method does not use `Eq` or `PartialEq` for comparison.
    ///
    /// `compare_exchange` takes two `Ordering` arguments to describe the memory
    /// ordering of this operation. The first describes the required ordering if
    /// the operation succeeds while the second describes the required ordering
    /// when the operation fails. The failure ordering can't be `Release` or
    /// `AcqRel` and must be equivalent or weaker than the success ordering.
    ///
    /// Calls with a success ordering weaker than `AcqRel` or failure ordering
    /// weaker than `Acquire` will be performed with those orderings.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::Ordering;
    /// use atomic_ref::AtomicRef;
    ///
    /// static VALUE: i32 = 10;
    /// static OTHER_VALUE: i32 = 20;
    ///
    /// let some_ptr = AtomicRef::new(Some(&VALUE));
    /// let value = some_ptr.compare_exchange(Some(&OTHER_VALUE), None,
    ///                                       Ordering::SeqCst, Ordering::Acquire);
    /// ```
    pub fn compare_exchange(
        &self,
        current: Option<&'a T>,
        new: Option<&'a T>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Option<&'a T>, Option<&'a T>> {
        unsafe {
            match self.data.compare_exchange(
                from_opt(current),
                from_opt(new),
                enforce_swap_ordering(success),
                enforce_load_ordering(failure),
            ) {
                Ok(p) => Ok(to_opt(p)),
                Err(p) => Err(to_opt(p)),
            }
        }
    }

    /// Stores a value into the pointer if the current value is the same as the `current` value.
    ///
    /// Unlike `compare_exchange`, this function is allowed to spuriously fail even when the
    /// comparison succeeds, which can result in more efficient code on some platforms. The
    /// return value is a result indicating whether the new value was written and containing the
    /// previous value.
    ///
    /// `compare_exchange_weak` takes two `Ordering` arguments to describe the memory
    /// ordering of this operation. The first describes the required ordering if the operation
    /// succeeds while the second describes the required ordering when the operation fails. The
    /// failure ordering can't be `Release` or `AcqRel` and must be equivalent or weaker than the
    /// success ordering.
    ///
    /// Calls with a success ordering weaker than `AcqRel` or failure ordering
    /// weaker than `Acquire` will be performed with those orderings.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::Ordering;
    /// use atomic_ref::AtomicRef;
    ///
    /// static VALUE: i32 = 10;
    /// static OTHER_VALUE: i32 = 20;
    ///
    /// let some_ptr = AtomicRef::new(Some(&VALUE));
    ///
    /// let mut old = some_ptr.load(Ordering::Acquire);
    /// loop {
    ///     match some_ptr.compare_exchange_weak(old, Some(&VALUE),
    ///                                          Ordering::SeqCst, Ordering::Acquire) {
    ///         Ok(_) => break,
    ///         Err(x) => old = x,
    ///     }
    /// }
    /// ```
    pub fn compare_exchange_weak(
        &self,
        current: Option<&'a T>,
        new: Option<&'a T>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Option<&'a T>, Option<&'a T>> {
        unsafe {
            match self.data.compare_exchange_weak(
                from_opt(current),
                from_opt(new),
                enforce_swap_ordering(success),
                enforce_load_ordering(failure),
            ) {
                Ok(p) => Ok(to_opt(p)),
                Err(p) => Err(to_opt(p)),
            }
        }
    }
}

impl<'a, T: fmt::Debug> fmt::Debug for AtomicRef<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.load(Ordering::SeqCst), f)
    }
}

impl<'a, T> Default for AtomicRef<'a, T> {
    fn default() -> AtomicRef<'a, T> {
        AtomicRef::new(None)
    }
}

impl<'a, T> From<Option<&'a T>> for AtomicRef<'a, T> {
    fn from(other: Option<&'a T>) -> AtomicRef<'a, T> {
        AtomicRef::new(other)
    }
}

#[cfg(test)]
mod tests {
    use super::AtomicRef;
    use core::sync::atomic::Ordering;

    static FOO: AtomicRef<i32> = AtomicRef::new(None);

    static A: i32 = 10;

    #[test]
    fn it_works() {
        assert!(FOO.load(Ordering::SeqCst) == None);
        FOO.store(Some(&A), Ordering::SeqCst);
        assert!(FOO.load(Ordering::SeqCst) == Some(&A));
        assert!(FOO.load(Ordering::SeqCst).unwrap() as *const _ == &A as *const _);
    }
}
