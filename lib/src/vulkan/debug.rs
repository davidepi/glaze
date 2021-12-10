use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Validation layers enabled when compiling in debug mode
pub const DEFAULT_VALIDATIONS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
/// Force validation layers in release mode
pub const FORCE_VALIDATIONS: bool = false;

/// Struct holding a list of validation layers (in CString form)
/// Mostly used as a bridge between the Rust-strings provided here and C-strings required by Vulkan
pub struct ValidationLayers {
    /// All the validation layer names as CStrings
    names: Vec<CString>,
}

impl ValidationLayers {
    /// Creates a new ValidationLayers struct from a list of validation layers names
    pub fn new(names: &[&str]) -> ValidationLayers {
        if cfg!(debug_assertions) || FORCE_VALIDATIONS {
            let names = names
                .iter()
                .map(|x| CString::new(*x).unwrap())
                .collect::<Vec<_>>();
            ValidationLayers { names }
        } else {
            ValidationLayers {
                names: Vec::with_capacity(0),
            }
        }
    }

    /// Returns the validation layers enabled by default
    pub fn application_default() -> ValidationLayers {
        Self::new(&DEFAULT_VALIDATIONS)
    }

    /// Returns an array of pointers to each enabled validation layer
    pub fn as_ptr(&self) -> Vec<*const i8> {
        if cfg!(debug_assertions) || FORCE_VALIDATIONS {
            self.names.iter().map(|x| x.as_ptr()).collect::<Vec<_>>()
        } else {
            Vec::new()
        }
    }

    /// Checks whether the validation layers are supported or not
    pub fn check_support(&self, entry: &ash::Entry) -> bool {
        if cfg!(debug_assertions) || FORCE_VALIDATIONS {
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

/// Enables logging of Vulkan events
#[cfg(any(debug_assertions, FORCE_VALIDATIONS))]
pub(super) mod logger {
    use ash::vk;
    use std::ffi::{c_void, CStr};
    use std::ptr;

    /// Prints vulkan debug messages, grouped by severity
    pub unsafe extern "system" fn debug_print_callback(
        severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        mtype: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _: *mut c_void,
    ) -> vk::Bool32 {
        let typestr = match mtype {
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[GENERAL]",
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[PERFORMANCE]",
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[VALIDATION]",
            _ => "[UNKNOWN]",
        };
        let message = CStr::from_ptr((*p_callback_data).p_message);
        match severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
                log::trace!("{}{}", typestr, message.to_str().unwrap())
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
                log::warn!("{}{}", typestr, message.to_str().unwrap())
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
                log::error!("{}{}", typestr, message.to_str().unwrap())
            }
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
                log::info!("{}{}", typestr, message.to_str().unwrap())
            }
            _ => println!("{}{}", typestr, message.to_str().unwrap()),
        };
        vk::FALSE
    }

    /// Logger for vulkan events
    pub struct VkDebugLogger {
        debug_messenger: vk::DebugUtilsMessengerEXT,
        debug_loader: ash::extensions::ext::DebugUtils,
    }

    impl VkDebugLogger {
        /// Creates a new vulkan logger for the given instance.
        /// Logs only ERROR and WARNING messages.
        pub fn new(entry: &ash::Entry, instance: &ash::Instance) -> VkDebugLogger {
            let debug_loader = ash::extensions::ext::DebugUtils::new(entry, instance);
            let ci = vk::DebugUtilsMessengerCreateInfoEXT {
                s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                p_next: ptr::null(),
                flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                pfn_user_callback: Some(debug_print_callback),
                p_user_data: ptr::null_mut(),
            };
            let debug_messenger = unsafe {
                debug_loader
                    .create_debug_utils_messenger(&ci, None)
                    .expect("Failed to create debug messenger")
            };
            VkDebugLogger {
                debug_messenger,
                debug_loader,
            }
        }
    }

    impl Drop for VkDebugLogger {
        fn drop(&mut self) {
            unsafe {
                self.debug_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
        }
    }
}
