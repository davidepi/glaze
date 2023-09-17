use crate::graphics::error::{GraphicError, ErrorCategory};

impl From<ash::vk::Result> for GraphicError {
    fn from(value: ash::vk::Result) -> Self {
        match value {
                ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY  => Self::new(ErrorCategory::HostMemory, "A host memory allocation has failed."),
                ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY  => Self::new(ErrorCategory::DeviceMemory, "A device memory allocation has failed."),
                ash::vk::Result::ERROR_INITIALIZATION_FAILED  => Self::new(ErrorCategory::InitFailed, "Initialization of an object could not be completed for implementation-specific reasons."),
                ash::vk::Result::ERROR_DEVICE_LOST  => Self::new(ErrorCategory::DeviceLost, "The logical or physical device has been lost. See Lost Device"),
                ash::vk::Result::ERROR_MEMORY_MAP_FAILED  => Self::new(ErrorCategory::Uncategorized, "Mapping of a memory object has failed."),
                ash::vk::Result::ERROR_LAYER_NOT_PRESENT  => Self::new(ErrorCategory::UnsupportedFeature, "A requested layer is not present or could not be loaded."),
                ash::vk::Result::ERROR_EXTENSION_NOT_PRESENT  => Self::new(ErrorCategory::UnsupportedFeature, "A requested extension is not supported."),
                ash::vk::Result::ERROR_FEATURE_NOT_PRESENT  => Self::new(ErrorCategory::UnsupportedFeature, "A requested feature is not supported."),
                ash::vk::Result::ERROR_INCOMPATIBLE_DRIVER  => Self::new(ErrorCategory::Uncategorized, "The requested version of Vulkan is not supported by the driver or is otherwise incompatible for implementation-specific reasons."),
                ash::vk::Result::ERROR_TOO_MANY_OBJECTS  => Self::new(ErrorCategory::Uncategorized, "Too many objects of the type have already been created."),
                ash::vk::Result::ERROR_FORMAT_NOT_SUPPORTED  => Self::new(ErrorCategory::UnsupportedFeature, "A requested format is not supported on this device."),
                ash::vk::Result::ERROR_FRAGMENTED_POOL  => Self::new(ErrorCategory::Uncategorized, "A pool allocation has failed due to fragmentation of the pool’s memory. This must only be returned if no attempt to allocate host or device memory was made to accommodate the new allocation. This should be returned in preference to VK_ERROR_OUT_OF_POOL_MEMORY, but only if the implementation is certain that the pool allocation failure was due to fragmentation."),
                ash::vk::Result::ERROR_SURFACE_LOST_KHR  => Self::new(ErrorCategory::DeviceLost, "A surface is no longer available."),
                ash::vk::Result::ERROR_NATIVE_WINDOW_IN_USE_KHR  => Self::new(ErrorCategory::Uncategorized, "The requested window is already in use by Vulkan or another API in a manner which prevents it from being used again."),
                ash::vk::Result::ERROR_OUT_OF_DATE_KHR  => Self::new(ErrorCategory::Uncategorized, "A surface has changed in such a way that it is no longer compatible with the swapchain, and further presentation requests using the swapchain will fail. Applications must query the new surface properties and recreate their swapchain if they wish to continue presenting to the surface."),
                ash::vk::Result::ERROR_INCOMPATIBLE_DISPLAY_KHR  => Self::new(ErrorCategory::Uncategorized, "The display used by a swapchain does not use the same presentable image layout, or is incompatible in a way that prevents sharing an image."),
                ash::vk::Result::ERROR_INVALID_SHADER_NV  => Self::new(ErrorCategory::Uncategorized, "One or more shaders failed to compile or link. More details are reported back to the application via VK_EXT_debug_report if enabled."),
                ash::vk::Result::ERROR_OUT_OF_POOL_MEMORY  => Self::new(ErrorCategory::Uncategorized, "A pool memory allocation has failed. This must only be returned if no attempt to allocate host or device memory was made to accommodate the new allocation. If the failure was definitely due to fragmentation of the pool, VK_ERROR_FRAGMENTED_POOL should be returned instead."),
                ash::vk::Result::ERROR_INVALID_EXTERNAL_HANDLE  => Self::new(ErrorCategory::Uncategorized, "An external handle is not a valid handle of the specified type."),
                ash::vk::Result::ERROR_FRAGMENTATION  => Self::new(ErrorCategory::Uncategorized, "A descriptor pool creation has failed due to fragmentation."),
                ash::vk::Result::ERROR_INVALID_DEVICE_ADDRESS_EXT  => Self::new(ErrorCategory::Uncategorized, "A buffer creation failed because the requested address is not available."),
                ash::vk::Result::ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS  => Self::new(ErrorCategory::Uncategorized, "A buffer creation or memory allocation failed because the requested address is not available. A shader group handle assignment failed because the requested shader group handle information is no longer valid."),
                ash::vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT  => Self::new(ErrorCategory::Uncategorized, "An operation on a swapchain created with VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT failed as it did not have exclusive full-screen access. This may occur due to implementation-dependent reasons, outside of the application’s control."),
                ash::vk::Result::ERROR_VALIDATION_FAILED_EXT  => Self::new(ErrorCategory::Uncategorized, "A command failed because invalid usage was detected by the implementation or a validation-layer."),
                ash::vk::Result::ERROR_COMPRESSION_EXHAUSTED_EXT  => Self::new(ErrorCategory::Uncategorized, "An image creation failed because internal resources required for compression are exhausted. This must only be returned when fixed-rate compression is requested."),
                ash::vk::Result::ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR  => Self::new(ErrorCategory::UnsupportedFeature, "The requested VkImageUsageFlags are not supported."),
                ash::vk::Result::ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR  => Self::new(ErrorCategory::UnsupportedFeature, "The requested video picture layout is not supported."),
                ash::vk::Result::ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR  => Self::new(ErrorCategory::UnsupportedFeature, "A video profile operation specified via VkVideoProfileInfoKHR::videoCodecOperation is not supported."),
                ash::vk::Result::ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR  => Self::new(ErrorCategory::UnsupportedFeature, "Format parameters in a requested VkVideoProfileInfoKHR chain are not supported."),
                ash::vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR  => Self::new(ErrorCategory::UnsupportedFeature, "Codec-specific parameters in a requested VkVideoProfileInfoKHR chain are not supported."),
                ash::vk::Result::ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR  => Self::new(ErrorCategory::UnsupportedFeature, "The specified video Std header version is not supported."),
                ash::vk::Result::ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR  => Self::new(ErrorCategory::Uncategorized, "The specified Video Std parameters do not adhere to the syntactic or semantic requirements of the used video compression standard, or values derived from parameters according to the rules defined by the used video compression standard do not adhere to the capabilities of the video compression standard or the implementation."),
                ash::vk::Result::ERROR_INCOMPATIBLE_SHADER_BINARY_EXT  => Self::new(ErrorCategory::Uncategorized, "The provided binary shader code is not compatible with this device."),
                ash::vk::Result::ERROR_UNKNOWN  => Self::new(ErrorCategory::Uncategorized, "An unknown error has occurred; either the application has provided invalid input, or an implementation failure has occurred."),
                // these following errors should NEVER be raised because... they are not errors.
                // they should be handled beforehand by VkResult.
                ash::vk::Result::SUCCESS => Self::new(ErrorCategory::NotError, "Command successfully completed"),
                ash::vk::Result::NOT_READY => Self::new(ErrorCategory::NotError, "A fence or query has not yet completed"),
                ash::vk::Result::TIMEOUT => Self::new(ErrorCategory::NotError, "A wait operation has not completed in the specified time"),
                ash::vk::Result::EVENT_SET  => Self::new(ErrorCategory::NotError, "An event is signaled"),
                ash::vk::Result::EVENT_RESET  => Self::new(ErrorCategory::NotError, "An event is unsignaled"),
                ash::vk::Result::INCOMPLETE  => Self::new(ErrorCategory::NotError, "A return array was too small for the result"),
                ash::vk::Result::SUBOPTIMAL_KHR  => Self::new(ErrorCategory::NotError, "A swapchain no longer matches the surface properties exactly, but can still be used to present to the surface successfully."),
                ash::vk::Result::THREAD_IDLE_KHR  => Self::new(ErrorCategory::NotError, "A deferred operation is not complete but there is currently no work for this thread to do at the time of this call."),
                ash::vk::Result::THREAD_DONE_KHR  => Self::new(ErrorCategory::NotError, "A deferred operation is not complete but there is no work remaining to assign to additional threads."),
                ash::vk::Result::OPERATION_DEFERRED_KHR  => Self::new(ErrorCategory::NotError, "A deferred operation was requested and at least some of the work was deferred."),
                ash::vk::Result::OPERATION_NOT_DEFERRED_KHR  => Self::new(ErrorCategory::NotError, "A deferred operation was requested and no operations were deferred."),
                ash::vk::Result::PIPELINE_COMPILE_REQUIRED  => Self::new(ErrorCategory::NotError, "A requested pipeline creation would have required compilation, but the application requested compilation to not be performed."),
            _ => panic!("Unknown or unexpected error {:?}", value),
        }
    }
}
