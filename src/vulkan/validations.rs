use std::{
    collections::HashSet,
    ffi::{CStr, CString},
    os::raw::c_char,
    ptr,
};

pub const DEFAULT_VALIDATIONS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

pub struct ValidationLayers {
    names: Vec<CString>,
    pointers: Vec<*const i8>,
}

impl ValidationLayers {
    pub fn new(names: &[&str]) -> ValidationLayers {
        if cfg!(debug_assertions) {
            let names = names
                .iter()
                .map(|x| CString::new(*x).unwrap())
                .collect::<Vec<_>>();
            let pointers = names.iter().map(|x| x.as_ptr()).collect::<Vec<_>>();
            ValidationLayers { names, pointers }
        } else {
            ValidationLayers {
                names: Vec::with_capacity(0),
                pointers: Vec::with_capacity(0),
            }
        }
    }

    pub fn application_default() -> ValidationLayers {
        Self::new(&DEFAULT_VALIDATIONS)
    }

    pub fn as_ptr(&self) -> *const *const i8 {
        if cfg!(debug_assertions) {
            self.pointers.as_ptr()
        } else {
            ptr::null()
        }
    }

    pub fn len(&self) -> u32 {
        self.names.len() as u32
    }

    pub fn check_support(&self, entry: &ash::Entry) -> bool {
        if cfg!(debug_assertions) {
            let layer_properties = entry
                .enumerate_instance_layer_properties()
                .expect("Failed to enumerate layer properties");
            let available_layers = layer_properties
                .iter()
                .map(|x| &x.layer_name[..])
                .map(cchars_to_string)
                .collect::<HashSet<_>>();
            self.names
                .iter()
                .map(|name| name.to_str().unwrap().to_string())
                .fold(!layer_properties.is_empty(), |acc, req_layer| {
                    acc && available_layers.contains(&req_layer)
                })
        } else {
            true
        }
    }
}

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
