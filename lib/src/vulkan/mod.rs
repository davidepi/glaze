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
mod raytracer;
#[cfg(feature = "vulkan-interactive")]
mod renderer;
#[cfg(feature = "vulkan-interactive")]
mod renderpass;
mod scene;
mod surface;
#[cfg(feature = "vulkan-interactive")]
mod swapchain;
#[cfg(feature = "vulkan-interactive")]
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
pub use self::raytracer::RayTraceRenderer;

#[cfg(feature = "vulkan-interactive")]
pub use self::instance::PresentInstance;
#[cfg(feature = "vulkan-interactive")]
pub use self::renderer::RealtimeRenderer;
#[cfg(feature = "vulkan-interactive")]
pub use self::scene::VulkanScene;
