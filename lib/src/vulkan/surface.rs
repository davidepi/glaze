use ash::vk;
#[cfg(feature = "vulkan-interactive")]
use std::ptr;
#[cfg(feature = "vulkan-interactive")]
use winit::window::Window;

/// A surface used to draw to.
pub struct Surface {
    pub surface: vk::SurfaceKHR,
    pub loader: ash::extensions::khr::Surface,
}

impl Surface {
    /// Creates a Surface from a Window.
    #[cfg(feature = "vulkan-interactive")]
    pub fn new(entry: &ash::Entry, instance: &ash::Instance, window: &Window) -> Self {
        let surface =
            unsafe { create_surface(entry, instance, window) }.expect("Failed to create surface");
        let surface_loader = ash::extensions::khr::Surface::new(entry, instance);
        Surface {
            surface,
            loader: surface_loader,
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.surface, None);
        }
    }
}

/// Creates a surface from a Window (Windows/macOS/Linux(X11)).
#[cfg(feature = "vulkan-interactive")]
unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    #[cfg(target_os = "macos")]
    {
        use cocoa::appkit::{NSView, NSWindow};
        use metal::CoreAnimationLayer;
        use std::mem;
        use std::os::raw::c_void;
        use winit::platform::macos::WindowExtMacOS;

        let wnd: cocoa::base::id = mem::transmute(window.ns_window());

        let layer = CoreAnimationLayer::new();

        layer.set_edge_antialiasing_mask(0);
        layer.set_presents_with_transaction(false);
        layer.remove_all_animations();

        let view = wnd.contentView();

        layer.set_contents_scale(view.backingScaleFactor());
        view.setLayer(mem::transmute(layer.as_ref()));
        view.setWantsLayer(cocoa::base::YES);

        let create_info = vk::MacOSSurfaceCreateInfoMVK {
            s_type: vk::StructureType::MACOS_SURFACE_CREATE_INFO_MVK,
            p_next: ptr::null(),
            flags: Default::default(),
            p_view: window.ns_view() as *const c_void,
        };

        let macos_surface_loader = ash::extensions::mvk::MacOSSurface::new(entry, instance);
        macos_surface_loader.create_mac_os_surface(&create_info, None)
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::raw::c_void;
        use winapi::shared::windef::HWND;
        use winapi::um::libloaderapi::GetModuleHandleW;
        use winit::platform::windows::WindowExtWindows;

        let hwnd = window.hwnd() as HWND;
        let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
        let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
            s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: Default::default(),
            hinstance,
            hwnd: hwnd as *const c_void,
        };
        let win32_surface_loader = ash::extensions::khr::Win32Surface::new(entry, instance);
        win32_surface_loader.create_win32_surface(&win32_create_info, None)
    }
    #[cfg(target_os = "linux")]
    {
        use winit::platform::unix::WindowExtUnix;

        let x11_display = window.xlib_display().unwrap();
        let x11_window = window.xlib_window().unwrap();
        let x11_create_info = vk::XlibSurfaceCreateInfoKHR {
            s_type: vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: Default::default(),
            window: x11_window as vk::Window,
            dpy: x11_display as *mut vk::Display,
        };
        let xlib_surface_loader = ash::extensions::khr::XlibSurface::new(entry, instance);
        xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
    }
}
