use std::{
    collections::HashSet,
    ffi::{CStr, CString},
    os::raw::c_char,
};

pub(super) struct ValidationsRequested {
    names: Vec<CString>,
    pointers: Vec<*const i8>,
}

impl ValidationsRequested {
    pub(super) fn new(names: &[&str]) -> ValidationsRequested {
        let names = names
            .iter()
            .map(|x| CString::new(*x).unwrap())
            .collect::<Vec<_>>();
        let pointers = names.iter().map(|x| x.as_ptr()).collect::<Vec<_>>();
        ValidationsRequested { names, pointers }
    }

    pub(super) fn layers_ptr(&self) -> &[*const i8] {
        &self.pointers
    }

    pub(super) fn check_support(&self, entry: &ash::Entry) -> bool {
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
    }
}

pub(super) fn cchars_to_string(cchars: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = cchars.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert C string to Rust String")
        .to_owned()
}
