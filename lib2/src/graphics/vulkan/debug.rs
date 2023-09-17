use super::util::cchars_to_string;
use std::collections::HashSet;
use std::ffi::CString;

/// Validation layers enabled when compiling in debug mode
pub const DEFAULT_VALIDATIONS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

/// Struct holding a list of validation layers (in CString form)
/// Mostly used as a bridge between the Rust-strings provided here and C-strings required by Vulkan
pub struct ValidationLayers {
    /// All the validation layer names as CStrings
    names: Vec<CString>,
}

impl ValidationLayers {
    /// Creates a new ValidationLayers struct from a list of validation layers names
    pub fn new(names: &[&str]) -> ValidationLayers {
        let names = names
            .iter()
            .map(|x| CString::new(*x).unwrap())
            .collect::<Vec<_>>();
        ValidationLayers { names }
    }

    /// Returns the validation layers enabled by default
    pub fn application_default() -> ValidationLayers {
        Self::new(&DEFAULT_VALIDATIONS)
    }

    /// Returns the names of the validation layers as CString
    pub fn names(&self) -> &[CString] {
        self.names.as_ref()
    }

    /// Checks whether the validation layers are supported or not
    pub fn check_support(&self, entry: &ash::Entry) -> bool {
        let layer_properties = entry.enumerate_instance_layer_properties();
        if let Ok(layer_properties) = layer_properties {
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
            false
        }
    }
}

/// Enables logging of Vulkan events
#[cfg(debug_assertions)]
pub mod logger {
    use ash::vk;
    use std::ffi::{c_void, CStr};
    use std::ptr;

    use crate::graphics::error::GraphicError;

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
    pub struct VulkanDebugLogger {
        debug_messenger: vk::DebugUtilsMessengerEXT,
        debug_loader: ash::extensions::ext::DebugUtils,
    }

    impl VulkanDebugLogger {
        /// Creates a new vulkan logger for the given instance.
        /// Logs only ERROR and WARNING messages.
        pub fn new(
            entry: &ash::Entry,
            instance: &ash::Instance,
        ) -> Result<VulkanDebugLogger, GraphicError> {
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
            let debug_messenger = unsafe { debug_loader.create_debug_utils_messenger(&ci, None)? };
            Ok(VulkanDebugLogger {
                debug_messenger,
                debug_loader,
            })
        }
    }

    impl Drop for VulkanDebugLogger {
        fn drop(&mut self) {
            unsafe {
                self.debug_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
        }
    }
}
