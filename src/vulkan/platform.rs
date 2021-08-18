use std::{collections::HashSet, ffi::CStr, iter::FromIterator};

pub fn required_extension_names() -> HashSet<&'static CStr> {
    let retval;
    #[cfg(target_os = "macos")]
    {
        retval = [
            ash::extensions::khr::Surface::name(),
            ash::extensions::mvk::MacOSSurface::name(),
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name(),
        ]
    }
    #[cfg(target_os = "windows")]
    {
        retval = [
            ash::extensions::khr::Surface::name(),
            ash::extensions::khr::Win32Surface::name(),
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name(),
        ]
    }
    #[cfg(target_os = "linux")]
    {
        retval = [
            ash::extensions::khr::Surface::name(),
            ash::extensions::khr::XlibSurface::name(),
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name(),
        ]
    }
    //FIXME: after moving to edition 2021 remove .cloned() (Array::IntoIter stabilization)
    HashSet::from_iter(retval.into_iter().cloned())
}
