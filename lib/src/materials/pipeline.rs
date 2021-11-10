use crate::geometry::Vertex;
use ash::vk;
use cgmath::Matrix4;
use std::ffi::CString;
use std::ptr;

pub struct Pipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl Pipeline {
    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_pipeline(self.pipeline, None);
        }
    }
}

pub struct PipelineBuilder {
    pub shaders: Vec<(Vec<u8>, CString, vk::ShaderStageFlags)>,
    pub vertex_input: vk::PipelineVertexInputStateCreateInfo,
    pub input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    pub rasterizer: vk::PipelineRasterizationStateCreateInfo,
    pub multisampling: vk::PipelineMultisampleStateCreateInfo,
    pub blending_settings: Vec<vk::PipelineColorBlendAttachmentState>,
    pub blending: (Option<vk::LogicOp>, [f32; 4]),
}

impl PipelineBuilder {
    pub fn push_shader(&mut self, shader: &[u8], func: &'static str, stage: vk::ShaderStageFlags) {
        self.shaders.push((
            shader.to_vec(),
            CString::new(func.as_bytes()).unwrap(),
            stage,
        ));
    }

    pub fn build(
        self,
        device: &ash::Device,
        renderpass: vk::RenderPass,
        viewports: &[vk::Viewport],
        scissors: &[vk::Rect2D],
        set_layout: &[vk::DescriptorSetLayout],
    ) -> Pipeline {
        let shader_stages = self
            .shaders
            .iter()
            .map(|(shader, func, stage)| vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                stage: *stage,
                module: create_shader_module(device, &shader),
                p_name: func.as_c_str().as_ptr(),
                p_specialization_info: ptr::null(),
            })
            .collect::<Vec<_>>();
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
        };
        let color_blend = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: if self.blending.0.is_some() {
                vk::TRUE
            } else {
                vk::FALSE
            },
            logic_op: self.blending.0.unwrap_or(vk::LogicOp::COPY),
            attachment_count: self.blending_settings.len() as u32,
            p_attachments: self.blending_settings.as_ptr(),
            blend_constants: self.blending.1,
        };
        let push_constant = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: std::mem::size_of::<Matrix4<f32>>() as u32,
        }];
        let pipeline_layout = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: set_layout.len() as u32,
            p_set_layouts: set_layout.as_ptr(),
            push_constant_range_count: push_constant.len() as u32,
            p_push_constant_ranges: push_constant.as_ptr(),
        };
        let layout = unsafe { device.create_pipeline_layout(&pipeline_layout, None) }
            .expect("Failed to create Pipeline Layout");
        let create_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &self.vertex_input,
            p_input_assembly_state: &self.input_assembly,
            p_tessellation_state: ptr::null(),
            p_viewport_state: &viewport_state,
            p_rasterization_state: &self.rasterizer,
            p_multisample_state: &self.multisampling,
            p_depth_stencil_state: ptr::null(),
            p_color_blend_state: &color_blend,
            p_dynamic_state: ptr::null(),
            layout,
            render_pass: renderpass,
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: 0,
        };
        let pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
        }
        .expect("Failed to create Graphics Pipeline")[0];
        shader_stages
            .iter()
            .for_each(|ci| destroy_shader_module(device, ci.module));
        Pipeline { pipeline, layout }
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        let shaders = Vec::with_capacity(2);
        let binding_descriptions = Vertex::binding_descriptions();
        let attribute_descriptions = Vertex::attribute_descriptions();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            vertex_binding_description_count: binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: binding_descriptions.as_ptr(),
            vertex_attribute_description_count: attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
        };
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
        };
        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };
        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: Default::default(),
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 1.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
        };
        let blending_settings = vec![vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::all(),
        }];
        let blending = (None, [0.0; 4]);
        PipelineBuilder {
            shaders,
            vertex_input,
            input_assembly,
            rasterizer,
            multisampling,
            blending_settings,
            blending,
        }
    }
}

fn create_shader_module(device: &ash::Device, shader: &[u8]) -> vk::ShaderModule {
    let shader_module_create_info = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ShaderModuleCreateFlags::empty(),
        code_size: shader.len(),
        p_code: shader.as_ptr() as *const u32,
    };
    unsafe {
        device
            .create_shader_module(&shader_module_create_info, None)
            .expect("Failed to create shader module")
    }
}

fn destroy_shader_module(device: &ash::Device, shader_module: vk::ShaderModule) {
    unsafe {
        device.destroy_shader_module(shader_module, None);
    }
}
