use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
use std::marker::PhantomData;
use std::fmt;
use std::default::Default;

#[repr(C)]
pub struct AtomicRef<'a, T: 'a> {
    data: AtomicUsize,
    _marker: PhantomData<&'a T>,
}

/// We cannot have an ATOMIC_REF_INIT method, as we can't make a const which is
/// templated over a type
///
/// Instead, we define it for a specific type (in this case &'static u8), and
/// define a macro static_atomic_ref! which uses unsafe code to allow the
/// creation of other types based on this layout.
pub const ATOMIC_U8_REF_INIT: AtomicRef<'static, u8> = AtomicRef {
    data: ATOMIC_USIZE_INIT,
    _marker: PhantomData,
};

#[macro_export]
macro_rules! static_atomic_ref {
    ($(#[$attr:meta])* static $N:ident : AtomicRef<$T:ty>; $($t:tt)*) => {
        static_atomic_ref!(@PRIV, $(#[$attr])* static $N : $T; $($t)*);
    };
    ($(#[$attr:meta])* pub static $N:ident : AtomicRef<$T:ty>; $($t:tt)*) => {
        static_atomic_ref!(@PUB, $(#[$attr])* static $N : $T; $($t)*);
    };
    (@$VIS:ident, $(#[$attr:meta])* static $N:ident : $T:ty; $($t:tt)*) => {
        static_atomic_ref!(@MAKE TY, $VIS, $(#[$attr])*, $N);
        impl ::std::ops::Deref for $N {
            type Target = $crate::AtomicRef<'static, $T>;
            #[allow(unsafe_code)]
            fn deref<'a>(&'a self) -> &'a $crate::AtomicRef<'static, $T> {
                unsafe { ::std::mem::transmute(&self._ref) }
            }
        }
        static_atomic_ref!($($t)*);
    };
    (@MAKE TY, PUB, $(#[$attr:meta])*, $N:ident) => {
        #[allow(missing_copy_implementations)]
        #[allow(non_camel_case_types)]
        #[allow(dead_code)]
        $(#[$attr])*
        pub struct $N { _ref: $crate::AtomicRef<'static, u8> }
        #[doc(hidden)]
        pub static $N: $N = $N { _ref: $crate::ATOMIC_U8_REF_INIT };
    };
    (@MAKE TY, PRIV, $(#[$attr:meta])*, $N:ident) => {
        #[allow(missing_copy_implementations)]
        #[allow(non_camel_case_types)]
        #[allow(dead_code)]
        $(#[$attr])*
        struct $N { _ref: $crate::AtomicRef<'static, u8> }
        #[doc(hidden)]
        static $N: $N = $N { _ref: $crate::ATOMIC_U8_REF_INIT };
    };
    () => ();
}

fn from_opt<'a, T>(p: Option<&'a T>) -> usize {
    match p {
        Some(p) => p as *const T as usize,
        None => 0,
    }
}

unsafe fn to_opt<'a, T>(p: usize) -> Option<&'a T> {
    (p as *const T).as_ref()
}

impl<'a, T> AtomicRef<'a, T> {
    pub fn new(p: Option<&'a T>) -> AtomicRef<'a, T> {
        AtomicRef {
            data: AtomicUsize::new(from_opt(p)),
            _marker: PhantomData,
        }
    }

    pub fn load(&self, ordering: Ordering) -> Option<&'a T> {
        unsafe {
            to_opt(self.data.load(ordering))
        }
    }

    pub fn store(&self, ptr: Option<&'a T>, order: Ordering) {
        self.data.store(from_opt(ptr), order)
    }

    pub fn swap(&self, p: Option<&'a T>, order: Ordering) -> Option<&'a T> {
        unsafe {
            to_opt(self.data.swap(from_opt(p), order))
        }
    }

    pub fn compare_and_swap(&self, current: Option<&'a T>, new: Option<&'a T>, order: Ordering)
                            -> Option<&'a T> {
        unsafe {
            to_opt(self.data.compare_and_swap(from_opt(current), from_opt(new), order))
        }
    }

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
    use std::sync::atomic::Ordering;

    static_atomic_ref! {
        static FOO: AtomicRef<i32>;
    }

    static A: i32 = 10;

    #[test]
    fn it_works() {
        assert!(FOO.load(Ordering::SeqCst) == None);
        FOO.store(Some(&A), Ordering::SeqCst);
        assert!(FOO.load(Ordering::SeqCst) == Some(&A));
        assert!(FOO.load(Ordering::SeqCst).unwrap() as *const _ == &A as *const _);
    }
}
