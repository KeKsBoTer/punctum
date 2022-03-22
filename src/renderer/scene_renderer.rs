use std::sync::Arc;

use bytemuck::Zeroable;
use cgmath::{Matrix4, SquareMatrix};
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
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, PartialStateMode, Pipeline, PipelineBindPoint, StateMode,
    },
    render_pass::Subpass,
};

use crate::{camera::Camera, pointcloud::PointCloud, scene::Scene, vertex::Vertex};

use super::Frame;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/pointcloud.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
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

    uniform_buffer_pool: Arc<CpuBufferPool<vs::ty::UniformData, Arc<StdMemoryPool>>>,
    uniform_buffer: Arc<CpuBufferPoolSubbuffer<vs::ty::UniformData, Arc<StdMemoryPool>>>,
}

impl PointCloudRenderer {
    pub fn new(device: Arc<Device>, subpass: Subpass) -> Self {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pool = Arc::new(CpuBufferPool::new(device.clone(), BufferUsage::all()));

        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState {
                topology: PartialStateMode::Fixed(PrimitiveTopology::PointList),
                primitive_restart_enable: StateMode::Fixed(false),
            })
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(subpass)
            .build(device.clone())
            .unwrap();

        PointCloudRenderer {
            pipeline: pipeline,
            uniform_buffer_pool: pool.clone(),
            uniform_buffer: pool.next(vs::ty::UniformData::zeroed()).unwrap(),
        }
    }

    pub fn render_point_cloud(
        &self,
        queue: Arc<Queue>,
        point_cloud: &PointCloud,
        viewport: Viewport,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        // TODO move viewport to pipeline creation
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

        builder
            .set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .bind_vertex_buffers(0, point_cloud.buffer.clone())
            .draw(point_cloud.buffer.len() as u32, 1, 0, 0)
            .unwrap();

        Arc::new(builder.build().unwrap())
    }

    fn set_camera(&mut self, camera: Camera) {
        let uniform_data = vs::ty::UniformData {
            world: Matrix4::identity().into(),
            view: camera.view().clone().into(),
            proj: camera.projection().clone().into(),
        };
        self.uniform_buffer = self.uniform_buffer_pool.next(uniform_data).unwrap();
    }

    pub fn render_to_frame(&mut self, queue: Arc<Queue>, scene: &Scene, frame: &mut Frame) {
        self.set_camera(scene.camera);
        let cb =
            self.render_point_cloud(queue.clone(), scene.point_cloud(), frame.viewport().clone());
        frame.render(queue, cb);
    }
}
