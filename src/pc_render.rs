use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    device::Queue,
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

use crate::vertex::{self, Vertex};

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/vert.glsl"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/frag.glsl"
    }
}

pub struct PointCloudRenderer {
    queue: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
}

impl PointCloudRenderer {
    pub fn new(queue: Arc<Queue>, subpass: Subpass) -> Self {
        let vertex1 = vertex::Vertex {
            position: [-0.5, -0.5, 1.],
        };
        let vertex2 = vertex::Vertex {
            position: [0.0, 0.5, 1.],
        };
        let vertex3 = vertex::Vertex {
            position: [0.5, -0.25, 1.],
        };

        let device = queue.device();

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            vec![vertex1, vertex2, vertex3].into_iter(),
        )
        .unwrap();

        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<vertex::Vertex>())
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

        PointCloudRenderer {
            queue: queue.clone(),
            pipeline: pipeline,
            vertex_buffer: vertex_buffer,
        }
    }

    pub fn draw(&self, viewport_dimensions: [u32; 2]) -> Arc<SecondaryAutoCommandBuffer> {
        // let layout = self.pipeline.layout();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.queue.device().clone(),
            self.queue.family(),
            CommandBufferUsage::MultipleSubmit,
            self.pipeline.subpass().clone(),
        )
        .unwrap();

        builder
            .set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();

        Arc::new(builder.build().unwrap())
    }
}
