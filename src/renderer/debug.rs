use std::sync::{Arc, RwLock};

use nalgebra::Vector3;
use vulkano::{
    buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
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

use crate::{Octree, Vertex, Viewport};

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/lines.frag"
    }
}

mod vs {

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/lines.vert",

        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy,Pod, Zeroable)]
        }
    }
}

pub struct OctreeDebugRenderer {
    pipeline: RwLock<Arc<GraphicsPipeline>>,

    fs: Arc<ShaderModule>,
    vs: Arc<ShaderModule>,
    subpass: Subpass,

    /// all octree vertices in one block of memory
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex<f32, f32>]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
}

impl OctreeDebugRenderer {
    pub fn new(
        device: Arc<Device>,
        subpass: Subpass,
        viewport: Viewport,
        octree: Arc<Octree<f32, f32>>,
    ) -> Self {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pipeline = build_pipeline::<Vertex<f32, f32>>(
            vs.clone(),
            fs.clone(),
            subpass.clone(),
            viewport,
            device.clone(),
        );

        let mut vertices: Vec<Vertex<f32, f32>> =
            Vec::with_capacity(octree.num_octants() as usize * 8);
        let mut indices = Vec::with_capacity(octree.num_octants() as usize * 24);
        let mut index = 0;
        // for bbox in octree.octant_bboxes() {
        for octant in octree.into_octant_iterator() {
            vertices.extend(octant.bbox.corners().map(|p| Vertex {
                position: p.cast(),
                color: Vector3::new(1.0, 1., 0.),
            }));
            indices.extend_from_slice(&[
                // front
                index + 0,
                index + 1,
                index + 1,
                index + 3,
                index + 3,
                index + 2,
                index + 2,
                index + 0,
                // back
                index + 4,
                index + 5,
                index + 5,
                index + 7,
                index + 7,
                index + 6,
                index + 6,
                index + 4,
                // left
                index + 0,
                index + 4,
                index + 2,
                index + 6,
                // right
                index + 1,
                index + 5,
                index + 3,
                index + 7,
            ]);
            index += 24;
        }
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::vertex_buffer(),
            false,
            vertices,
        )
        .unwrap();
        let index_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::index_buffer(),
            false,
            indices,
        )
        .unwrap();

        Self {
            pipeline: RwLock::new(pipeline),
            vs,
            fs,
            subpass,
            vertex_buffer,
            index_buffer,
        }
    }
    pub fn render(
        &self,
        uniforms: Arc<dyn BufferAccess>,
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
    ) {
        let pipeline = self.pipeline.read().unwrap();

        let set_layouts = pipeline.layout().set_layouts();

        let set = PersistentDescriptorSet::new(
            set_layouts.get(0).unwrap().clone(),
            [WriteDescriptorSet::buffer(0, uniforms)],
        )
        .unwrap();

        builder
            .bind_pipeline_graphics(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                [set.clone()].to_vec(),
            );

        builder.bind_index_buffer(self.index_buffer.clone());
        builder.bind_vertex_buffers(0, self.vertex_buffer.clone());
        builder
            .draw_indexed(
                self.index_buffer.into_buffer_slice().len() as u32,
                1,
                0,
                0,
                0,
            )
            .unwrap();
    }

    //pub fn frustum_culling<'a>(&self, _visible_octants: &Vec<OctreeIter<'a, f32, f32>>) {}

    pub fn set_viewport(&self, viewport: Viewport) {
        let mut pipeline = self.pipeline.write().unwrap();
        *pipeline = build_pipeline::<Vertex<f32, f32>>(
            self.vs.clone(),
            self.fs.clone(),
            self.subpass.clone(),
            viewport,
            pipeline.device().clone(),
        );
    }
}

fn build_pipeline<V: vulkano::pipeline::graphics::vertex_input::Vertex>(
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    subpass: Subpass,
    viewport: Viewport,
    device: Arc<Device>,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<V>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState {
            topology: PartialStateMode::Fixed(PrimitiveTopology::LineList),
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
