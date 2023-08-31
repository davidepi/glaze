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
