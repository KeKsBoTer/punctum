use crate::{camera::Camera, pointcloud::PointCloudGPU, vertex::Vertex, Viewport};
use nalgebra::Matrix4;
use std::sync::Arc;
use vulkano::{
    buffer::{cpu_pool::CpuBufferPoolSubbuffer, BufferUsage, CpuBufferPool, TypedBufferAccess},
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
};

mod vs {
    use bytemuck::{Pod, Zeroable};
    use nalgebra::Matrix4;

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/pointcloud.vert",
    }

    #[derive(Clone, Copy, Zeroable, Pod)]
    #[repr(C)]
    pub struct UniformData {
        pub world: [[f32; 4]; 4],
        pub view: [[f32; 4]; 4],
        pub proj: [[f32; 4]; 4],
        pub point_size: f32,
    }

    impl Default for UniformData {
        fn default() -> Self {
            Self {
                world: Matrix4::identity().into(),
                view: Matrix4::identity().into(),
                proj: Matrix4::identity().into(),
                point_size: 10.,
            }
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
    pipeline: Arc<GraphicsPipeline>,

    uniform_buffer_pool: Arc<CpuBufferPool<vs::UniformData, Arc<StdMemoryPool>>>,
    uniform_buffer: Arc<CpuBufferPoolSubbuffer<vs::UniformData, Arc<StdMemoryPool>>>,

    fs: Arc<ShaderModule>,
    vs: Arc<ShaderModule>,

    uniform_data: vs::UniformData,
}

impl PointCloudRenderer {
    pub fn new(device: Arc<Device>, subpass: Subpass, viewport: Viewport) -> Self {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pool = Arc::new(CpuBufferPool::new(device.clone(), BufferUsage::all()));

        let pipeline =
            PointCloudRenderer::build_pipeline(vs.clone(), fs.clone(), subpass, viewport, device);

        PointCloudRenderer {
            pipeline: pipeline,
            uniform_buffer_pool: pool.clone(),
            uniform_buffer: pool.next(vs::UniformData::default()).unwrap(),

            vs: vs,
            fs: fs,

            uniform_data: vs::UniformData::default(),
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
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
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
        // TODO dont recreate every time
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, self.uniform_buffer.clone())],
        )
        .unwrap();

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
                set.clone(),
            )
            .bind_vertex_buffers(0, pc_buffer.clone())
            .draw(pc_buffer.len() as u32, 1, 0, 0)
            .unwrap();

        Arc::new(builder.build().unwrap())
    }

    pub fn set_point_size(&mut self, point_size: f32) {
        self.uniform_data = vs::UniformData {
            point_size,
            ..self.uniform_data
        };
        self.uniform_buffer = self.uniform_buffer_pool.next(self.uniform_data).unwrap();
    }

    pub fn set_camera(&mut self, camera: &Camera) {
        self.uniform_data = vs::UniformData {
            world: Matrix4::identity().into(),
            view: camera.view().clone().into(),
            proj: camera.projection().clone().into(),
            ..self.uniform_data
        };
        self.uniform_buffer = self.uniform_buffer_pool.next(self.uniform_data).unwrap();
    }
}
