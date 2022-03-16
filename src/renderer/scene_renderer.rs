use std::{borrow::BorrowMut, sync::Arc};

use vulkano::{
    buffer::TypedBufferAccess,
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    device::{Device, Queue},
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, PartialStateMode, StateMode,
    },
    render_pass::Subpass,
};

use crate::{pointcloud::PointCloud, scene::Scene, vertex::Vertex};

use super::Frame;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/pointcloud.vert"
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
}

impl PointCloudRenderer {
    pub fn new(device: Arc<Device>, subpass: Subpass) -> Self {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState {
                topology: PartialStateMode::Fixed(PrimitiveTopology::PointList),
                primitive_restart_enable: StateMode::Fixed(false),
            })
            .render_pass(subpass)
            .build(device.clone())
            .unwrap();

        PointCloudRenderer { pipeline: pipeline }
    }

    pub fn draw(
        &self,
        queue: Arc<Queue>,
        point_cloud: &PointCloud,
        viewport: Viewport,
    ) -> Arc<SecondaryAutoCommandBuffer> {
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
            .bind_vertex_buffers(0, point_cloud.buffer.clone())
            .draw(point_cloud.buffer.len() as u32, 1, 0, 0)
            .unwrap();

        Arc::new(builder.build().unwrap())
    }
}

pub struct SceneRenderer {
    pc_renderer: PointCloudRenderer,
}

impl SceneRenderer {
    pub fn new(device: Arc<Device>, subpass: Subpass) -> Self {
        SceneRenderer {
            pc_renderer: PointCloudRenderer::new(device, subpass),
        }
    }

    pub fn render_to_frame(&self, queue: Arc<Queue>, scene: &Scene, frame: &mut Frame) {
        let cb =
            self.pc_renderer
                .draw(queue.clone(), scene.point_cloud(), frame.viewport().clone());
        frame.render(queue, cb);
    }
}
