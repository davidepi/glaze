use crate::math::Extent2D;
use ash::vk;
use num_traits::AsPrimitive;
use std::ffi::{c_char, CStr};

/// converts a CString to a Rust String
pub fn cchars_to_string(cchars: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = cchars.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert C string to Rust String")
        .to_owned()
}

pub fn is_wayland() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok()
}

impl<T: AsPrimitive<u32>> Extent2D<T> {
    pub fn to_vk(self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.x.as_(),
            height: self.y.as_(),
        }
    }
}
