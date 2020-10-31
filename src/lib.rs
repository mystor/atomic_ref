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
//! To store an atomic reference in a static variable, a the macro
//! `static_atomic_ref!` must be used. A static initializer like
//! `ATOMIC_REF_INIT` is not possible due to the need to be generic over any
//! reference target type.
//!
//! This type in static position is often used for lazy global initialization.
//!
//! `AtomicRef` may only contain `Sized` types, as unsized types have wide
//! pointers which cannot be atomically written to or read from.
//!
//!
//! # Examples
//!
//! Static logger state
//!
//! ```
//! #[macro_use]
//! extern crate atomic_ref;
//! use atomic_ref::AtomicRef;
//! use std::sync::atomic::Ordering;
//! use std::io::{stdout, Write};
//!
//! // Define the idea of a logger
//! trait Logger {
//!     fn log(&self, msg: &str) {}
//! }
//! struct LoggerInfo {
//!     logger: &'static (Logger + Sync)
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

use core::sync::atomic::{AtomicPtr, Ordering};
use core::marker::PhantomData;
use core::fmt;
use core::ptr::null_mut;
use core::default::Default;

/// A mutable Option<&'a, T> type which can be safely shared between threads.
#[repr(C)]
pub struct AtomicRef<'a, T: 'a> {
    data: AtomicPtr<T>,
    // Make `AtomicRef` invariant over `'a` and `T`
    _marker: PhantomData<Invariant<'a, T>>,
}

// Work-around for the construction of `PhantomData<&mut _>` requiring
// `#![feature(const_fn)]`
struct Invariant<'a, T: 'a>(&'a mut &'a mut T);

/// Re-export `core` for `static_atomic_ref!` (which may be used in a
/// non-`no_std` crate, where `core` is unavailable).
#[doc(hidden)]
pub use core::{mem as core_mem, ops as core_ops};

/// An internal helper function for converting `Option<&'a T>` values to
/// `*mut T` for storing in the `AtomicUsize`.
const fn from_opt<'a, T>(p: Option<&'a T>) -> *mut T {
    match p {
        Some(p) => p as *const T as *mut T,
        None => null_mut(),
    }
}

/// An internal helper function for converting `*mut T` values stored in the
/// `AtomicUsize` back into `Option<&'a T>` values.
unsafe fn to_opt<'a, T>(p: *mut T) -> Option<&'a T> {
    p.as_ref()
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

    /// Loads the value stored in the `AtomicRef`.
    ///
    /// `load` takes an `Ordering` argument which describes the memory ordering of this operation.
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
    /// assert_eq!(some_ref.load(Ordering::Relaxed), Some(&10));
    /// ```
    pub fn load(&self, ordering: Ordering) -> Option<&'a T> {
        unsafe {
            to_opt(self.data.load(ordering))
        }
    }

    /// Stores a value into the `AtomicRef`.
    ///
    /// `store` takes an `Ordering` argument which describes the memory ordering of this operation.
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
    /// some_ptr.store(Some(&VALUE), Ordering::Relaxed);
    /// ```
    pub fn store(&self, ptr: Option<&'a T>, order: Ordering) {
        self.data.store(from_opt(ptr), order)
    }

    /// Stores a value into the `AtomicRef`, returning the old value.
    ///
    /// `swap` takes an `Ordering` argument which describes the memory ordering of this operation.
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
    /// let value = some_ptr.swap(Some(&OTHER_VALUE), Ordering::Relaxed);
    /// ```
    pub fn swap(&self, p: Option<&'a T>, order: Ordering) -> Option<&'a T> {
        unsafe {
            to_opt(self.data.swap(from_opt(p), order))
        }
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
    /// let value = some_ptr.compare_and_swap(Some(&OTHER_VALUE), None, Ordering::Relaxed);
    /// ```
    pub fn compare_and_swap(&self, current: Option<&'a T>, new: Option<&'a T>, order: Ordering)
                            -> Option<&'a T> {
        unsafe {
            to_opt(self.data.compare_and_swap(from_opt(current), from_opt(new), order))
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
    ///                                       Ordering::SeqCst, Ordering::Relaxed);
    /// ```
    pub fn compare_exchange(&self, current: Option<&'a T>, new: Option<&'a T>,
                            success: Ordering, failure: Ordering)
                            -> Result<Option<&'a T>, Option<&'a T>> {
        unsafe {
            match self.data.compare_exchange(from_opt(current), from_opt(new),
                                             success, failure) {
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
    /// let mut old = some_ptr.load(Ordering::Relaxed);
    /// loop {
    ///     match some_ptr.compare_exchange_weak(old, Some(&VALUE),
    ///                                          Ordering::SeqCst, Ordering::Relaxed) {
    ///         Ok(_) => break,
    ///         Err(x) => old = x,
    ///     }
    /// }
    /// ```
    pub fn compare_exchange_weak(&self, current: Option<&'a T>, new: Option<&'a T>,
                                 success: Ordering, failure: Ordering)
                                 -> Result<Option<&'a T>, Option<&'a T>> {
        unsafe {
            match self.data.compare_exchange_weak(from_opt(current), from_opt(new),
                                                  success, failure) {
                Ok(p) => Ok(to_opt(p)),
                Err(p) => Err(to_opt(p)),
            }
        }
    }
}


impl<'a, T: fmt::Debug> fmt::Debug for AtomicRef<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("AtomicRef").field(&self.load(Ordering::SeqCst)).finish()
    }
}

impl<'a, T> Default for AtomicRef<'a, T> {
    fn default() -> AtomicRef<'a, T> {
        AtomicRef::new(None)
    }
}

#[cfg(test)]
mod tests {
    use core::sync::atomic::Ordering;
    use super::AtomicRef;

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
