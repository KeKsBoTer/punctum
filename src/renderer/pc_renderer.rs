use crate::{
    camera::{Camera, Projection},
    pointcloud::PointCloudGPU,
    vertex::Vertex,
    Viewport,
};
use nalgebra::Matrix4;
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuBufferPool, DeviceLocalBuffer, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    memory::pool::StdMemoryPool,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline, PartialStateMode, Pipeline, PipelineBindPoint, StateMode,
    },
    render_pass::Subpass,
    shader::ShaderModule,
    sync::{self, GpuFuture},
};

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/pointcloud.vert",

        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy,Pod, Zeroable,Default)]
        }
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/pointcloud.frag"
    }
}

pub struct PointCloudRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,

    pipeline: Arc<GraphicsPipeline>,
    set: Arc<PersistentDescriptorSet>,

    uniform_buffer_pool: Arc<CpuBufferPool<vs::ty::UniformData, Arc<StdMemoryPool>>>,
    uniform_buffer: Arc<DeviceLocalBuffer<vs::ty::UniformData>>,
    uniform_data: vs::ty::UniformData,
    fs: Arc<ShaderModule>,
    vs: Arc<ShaderModule>,
}

impl PointCloudRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass,
        viewport: Viewport,
    ) -> Self {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pool = Arc::new(CpuBufferPool::new(device.clone(), BufferUsage::all()));

        let pipeline = PointCloudRenderer::build_pipeline(
            vs.clone(),
            fs.clone(),
            subpass,
            viewport,
            device.clone(),
        );

        let uniform_buffer: Arc<DeviceLocalBuffer<vs::ty::UniformData>> = DeviceLocalBuffer::new(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            None,
        )
        .unwrap();

        let layout = pipeline.layout().set_layouts().get(0).unwrap();

        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
        )
        .unwrap();

        PointCloudRenderer {
            device,
            queue,
            pipeline,
            set,
            uniform_buffer_pool: pool,
            uniform_buffer,
            uniform_data: vs::ty::UniformData::default(),
            vs: vs,
            fs: fs,
        }
    }

    fn build_pipeline(
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
        subpass: Subpass,
        viewport: Viewport,
        device: Arc<Device>,
    ) -> Arc<GraphicsPipeline> {
        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex<f32, f32>>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState {
                topology: PartialStateMode::Fixed(PrimitiveTopology::PointList),
                primitive_restart_enable: StateMode::Fixed(false),
            })
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                viewport.into()
            ]))
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(subpass)
            .build(device.clone())
            .unwrap()
    }

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.pipeline = PointCloudRenderer::build_pipeline(
            self.vs.clone(),
            self.fs.clone(),
            self.pipeline.subpass().clone(),
            viewport,
            self.pipeline.device().clone(),
        );
    }

    pub fn render_point_cloud(
        &self,
        queue: Arc<Queue>,
        point_cloud: &PointCloudGPU,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            queue.device().clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
            self.pipeline.subpass().clone(),
        )
        .unwrap();

        let pc_buffer = point_cloud.gpu_buffer();

        builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.set.clone(),
            )
            .bind_vertex_buffers(0, pc_buffer.clone())
            .draw(pc_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        Arc::new(builder.build().unwrap())
    }

    fn update_uniforms(&mut self) {
        let new_uniform_buffer = self.uniform_buffer_pool.next(self.uniform_data).unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(new_uniform_buffer, self.uniform_buffer.clone())
            .unwrap();

        let cb = builder.build().unwrap();

        // TODO dont do wait here but in render
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    pub fn set_point_size(&mut self, point_size: u32) {
        self.uniform_data = vs::ty::UniformData {
            point_size,
            ..self.uniform_data
        };
        self.update_uniforms();
    }

    pub fn set_camera(&mut self, camera: &Camera<impl Projection>) {
        self.uniform_data = vs::ty::UniformData {
            world: Matrix4::identity().into(),
            view: camera.view().clone().into(),
            proj: camera.projection().clone().into(),
            ..self.uniform_data
        };
        self.update_uniforms();
    }
}
