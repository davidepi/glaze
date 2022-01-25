use super::cmd::CommandManager;
use super::descriptor::DescriptorSetManager;
use super::instance::Instance;
use super::scene::RayTraceScene;
use super::{AllocatedImage, Descriptor};
use crate::{
    ParsedScene, PresentInstance, RayTraceInstance, TextureFormat, TextureInfo, TextureLoaded,
    VulkanScene,
};
use ash::extensions::khr::AccelerationStructure as AccelerationLoader;
use ash::vk;
use std::ptr;
use std::sync::mpsc::Sender;
use std::sync::Arc;

pub const RAYTRACE_SPLIT_SIZE: u32 = 64;

pub struct RayTraceRenderer<T: Instance + Send + Sync> {
    scene: RayTraceScene,
    extent: vk::Extent2D,
    out_img: AllocatedImage,
    descriptor: Descriptor,
    dm: DescriptorSetManager,
    tcmdm: CommandManager,
    ccmdm: CommandManager,
    loader: Arc<AccelerationLoader>,
    instance: Arc<T>,
}

impl<T: Instance + Send + Sync> RayTraceRenderer<T> {
    pub fn new(
        instance: Arc<RayTraceInstance>,
        scene: Box<dyn ParsedScene>,
        width: u32,
        height: u32,
    ) -> Result<RayTraceRenderer<RayTraceInstance>, std::io::Error> {
        let device = instance.device();
        let compute = device.compute_queue();
        let mut ccmdm = CommandManager::new(device.logical_clone(), compute.idx, 15);
        let loader = Arc::new(AccelerationLoader::new(
            instance.instance(),
            device.logical(),
        ));
        let scene = RayTraceScene::new(instance.clone(), loader.clone(), scene, &mut ccmdm)?;
        let extent = vk::Extent2D { width, height };
        Ok(init_rt(instance, loader, ccmdm, scene, extent))
    }

    pub(crate) fn from_realtime(
        instance: Arc<PresentInstance>,
        scene: &mut VulkanScene,
        width: u32,
        height: u32,
    ) -> Result<RayTraceRenderer<PresentInstance>, std::io::Error> {
        let device = instance.device();
        let compute = device.compute_queue();
        let mut ccmdm = CommandManager::new(device.logical_clone(), compute.idx, 15);
        let loader = Arc::new(AccelerationLoader::new(
            instance.instance(),
            device.logical(),
        ));
        let mm = instance.allocator();
        let scene = RayTraceScene::from(loader.clone(), scene, mm, &mut ccmdm)?;
        let extent = vk::Extent2D { width, height };
        Ok(init_rt(instance, loader, ccmdm, scene, extent))
    }

    pub fn draw(mut self, channel: Sender<String>) -> TextureLoaded {
        // if the other end disconnected, this thread can die anyway, so unwrap()
        channel.send("Starting rendering".to_string()).unwrap();
        channel.send("Rendering finished".to_string()).unwrap();
        channel.send("Copying result".to_string()).unwrap();
        let out_image = copy_storage_to_output(
            self.instance.as_ref(),
            self.tcmdm.get_cmd_buffer(),
            &self.out_img,
            self.extent,
        );
        let out_info = TextureInfo {
            name: "RT out image".to_string(),
            format: TextureFormat::Rgba,
            width: self.extent.width as u16,
            height: self.extent.height as u16,
        };
        TextureLoaded {
            info: out_info,
            image: out_image,
        }
    }
}

fn copy_storage_to_output<T: Instance>(
    instance: &T,
    tcmd: vk::CommandBuffer,
    img: &AllocatedImage,
    extent: vk::Extent2D,
) -> AllocatedImage {
    // blit the final image to one with a better layout to be displayed
    // probably later I want to just COPY the image, perform manual exposure compensation,
    // and then blit to sRGB or 8bit format.
    //TODO: probably I want this to be R32G32B32A32_SFLOAT
    let retval = instance.allocator().create_image_gpu(
        "RT output",
        vk::Format::R8G8B8A8_SRGB,
        extent,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::ImageAspectFlags::COLOR,
        1,
    );
    let blit_subresource = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let subresource_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let blit_regions = [vk::ImageBlit {
        src_subresource: blit_subresource,
        src_offsets: [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: extent.width as i32,
                y: extent.height as i32,
                z: 1,
            },
        ],
        dst_subresource: blit_subresource,
        dst_offsets: [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: extent.width as i32,
                y: extent.height as i32,
                z: 1,
            },
        ],
    }];
    let barrier_transfer = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::empty(),
        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: retval.image,
        subresource_range,
    };
    let barrier_use = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: retval.image,
        subresource_range,
    };
    let device = instance.device();
    let transfer_queue = device.transfer_queue();
    let command = |vkdevice: &ash::Device, cmd: vk::CommandBuffer| unsafe {
        vkdevice.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_transfer],
        );
        vkdevice.cmd_blit_image(
            cmd,
            img.image,
            vk::ImageLayout::GENERAL,
            retval.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &blit_regions,
            vk::Filter::LINEAR,
        );
        vkdevice.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_use],
        );
    };
    let fence = device.immediate_execute(tcmd, transfer_queue, command);
    device.wait_completion(&[fence]);
    retval
}

fn init_rt<T: Instance + Send + Sync>(
    instance: Arc<T>,
    loader: Arc<AccelerationLoader>,
    ccmdm: CommandManager,
    scene: RayTraceScene,
    extent: vk::Extent2D,
) -> RayTraceRenderer<T> {
    const AVG_DESC: [(vk::DescriptorType, f32); 3] = [
        (vk::DescriptorType::UNIFORM_BUFFER, 1.0),
        (vk::DescriptorType::STORAGE_BUFFER, 1.0),
        (vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, 1.0),
    ];
    let device = instance.device();
    let transfer_queue = device.transfer_queue();
    let tcmdm = CommandManager::new(device.logical_clone(), transfer_queue.idx, 1);
    let mut dm = DescriptorSetManager::new(
        device.logical_clone(),
        &AVG_DESC,
        instance.desc_layout_cache(),
    );
    let vbinfo = vk::DescriptorBufferInfo {
        buffer: scene.vertex_buffer.buffer,
        offset: 0,
        range: scene.vertex_buffer.size,
    };
    let ibinfo = vk::DescriptorBufferInfo {
        buffer: scene.index_buffer.buffer,
        offset: 0,
        range: scene.index_buffer.size,
    };
    let out_img = instance.allocator().create_image_gpu(
        "RT out image",
        vk::Format::R32G32B32A32_SFLOAT,
        extent,
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::ImageAspectFlags::COLOR,
        1,
    );
    let outimg_descinfo = vk::DescriptorImageInfo {
        sampler: vk::Sampler::null(),
        image_view: out_img.image_view,
        image_layout: vk::ImageLayout::GENERAL,
    };
    let descriptor = dm
        .new_set()
        .bind_buffer(
            vbinfo,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::RAYGEN_KHR,
        )
        .bind_buffer(
            ibinfo,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::RAYGEN_KHR,
        )
        .bind_acceleration_structure(&scene.acc.tlas.accel, vk::ShaderStageFlags::RAYGEN_KHR)
        .bind_image(
            outimg_descinfo,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::RAYGEN_KHR,
        )
        .build();
    RayTraceRenderer {
        scene,
        extent,
        tcmdm,
        out_img,
        descriptor,
        dm,
        ccmdm,
        loader,
        instance,
    }
}

#[cfg(test)]
mod tests {
    use super::RayTraceRenderer;
    use crate::{parse, RayTraceInstance};
    use std::path::PathBuf;
    use std::sync::Arc;

    #[test]
    fn load_raytrace() {
        env_logger::init();
        if let Some(instance) = RayTraceInstance::new() {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("resources")
                .join("cube.glaze");
            let parsed = parse(path).unwrap();
            let _ = RayTraceRenderer::<RayTraceInstance>::new(Arc::new(instance), parsed, 2, 2)
                .unwrap();
        } else {
            // SKIPPED does not exists in cargo test...
        }
    }
}
