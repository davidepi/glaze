mod acceleration;
mod cmd;
mod debug;
mod descriptor;
mod device;
#[cfg(feature = "vulkan-interactive")]
mod imgui;
mod instance;
mod memory;
mod pipeline;
pub(crate) mod raytrace_structures;
mod raytracer;
#[cfg(feature = "vulkan-interactive")]
mod renderer;
#[cfg(feature = "vulkan-interactive")]
mod renderpass;
mod scene;
mod surface;
mod sync;

// used in the material module
pub use self::pipeline::{Pipeline, PipelineBuilder};
// used in the texture module
pub use self::descriptor::Descriptor;
// used in the texture module to export a texture (independently of the renderer)
pub use self::cmd::CommandManager;
pub use self::device::{DeviceInfo, UnfinishedExecutions};
pub use self::instance::{Instance, RayTraceInstance};
pub use self::memory::{export, AllocatedImage};
pub use self::raytracer::{Integrator, RayTraceRenderer};
pub use self::scene::RayTraceScene;

#[cfg(feature = "vulkan-interactive")]
pub use self::instance::PresentInstance;
#[cfg(feature = "vulkan-interactive")]
pub use self::renderer::RealtimeRenderer;
#[cfg(feature = "vulkan-interactive")]
pub use self::scene::RealtimeScene;

/// Number of frames prepared by the CPU while waiting for the GPU.
const FRAMES_IN_FLIGHT: usize = 2;

/// Macro used to include the shader contained inside the /shader directory as a `[u8; _]`.
///
/// Probably will not work outside this crate.
#[macro_export]
macro_rules! include_shader {
    ($shader_name : expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/shaders/", $shader_name, ".spv"))
    };
}

/// Reads a struct as a sequence of bytes
unsafe fn as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts((p as *const T) as *const u8, std::mem::size_of::<T>())
}
