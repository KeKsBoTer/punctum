use crate::{
    camera::{Camera, Projection},
    vertex::Vertex,
    Octree, PerspectiveCamera, Viewport,
};
use nalgebra::{vector, Matrix4, Vector3};
use std::sync::Arc;
use vulkano::{
    buffer::{cpu_pool::CpuBufferPoolSubbuffer, BufferUsage, CpuBufferPool},
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

use super::UniformBuffer;

mod vs {
    use bytemuck::{Pod, Zeroable};
    use nalgebra::Matrix4;

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/pointcloud.vert",
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    pub struct UniformData {
        pub world: Matrix4<f32>,
        pub view: Matrix4<f32>,
        pub proj: Matrix4<f32>,
        pub point_size: u32,
        pub znear: f32,
        pub zfar: f32,
    }

    unsafe impl Zeroable for UniformData {}
    unsafe impl Pod for UniformData {}

    impl Default for UniformData {
        fn default() -> Self {
            Self {
                world: Matrix4::identity().into(),
                view: Matrix4::identity().into(),
                proj: Matrix4::identity().into(),
                point_size: 1,
                znear: 0.01,
                zfar: 100.,
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

struct OctantBuffer<const T: usize> {
    length: u32,
    points: Arc<CpuBufferPoolSubbuffer<[Vertex<f32, f32>; T], Arc<StdMemoryPool>>>,
}

pub struct OctreeRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,

    pipeline: Arc<GraphicsPipeline>,
    set: Arc<PersistentDescriptorSet>,

    uniforms: UniformBuffer<vs::UniformData>,
    fs: Arc<ShaderModule>,
    vs: Arc<ShaderModule>,

    octree: Octree<f32, f32>,
    octants: Vec<OctantBuffer<512>>,
}

impl OctreeRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass,
        viewport: Viewport,
        octree: Octree<f32, f32>,
    ) -> Self {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pipeline =
            Self::build_pipeline(vs.clone(), fs.clone(), subpass, viewport, device.clone());

        let max_size = octree.size();
        let center = octree.center();
        let scale_size = 1000.;

        let world =
            Matrix4::new_scaling(scale_size / max_size) * Matrix4::new_translation(&-center.coords);
        // let world = Matrix4::identity();

        let uniforms = UniformBuffer::new(
            device.clone(),
            Some(vs::UniformData {
                world: world.into(),
                ..vs::UniformData::default()
            }),
        );

        let layout = pipeline.layout().set_layouts().get(0).unwrap();

        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniforms.buffer().clone())],
        )
        .unwrap();

        let pool: CpuBufferPool<[Vertex<f32, f32>; 512]> =
            CpuBufferPool::new(device.clone(), BufferUsage::vertex_buffer());

        println!("center: {:?}, size: {}", center, max_size);

        let octants = octree
            .into_iter()
            .map(|octant| {
                let mut data = [Vertex::<f32, f32>::default(); 512];
                for (i, p) in octant.data.iter().enumerate() {
                    data[i] = Vertex {
                        // TODO move this operation to the world matrix
                        position: p.position.into(), //((p.position - center) / max_size * scale_size).into(),
                        color: p.color,
                    };
                }
                OctantBuffer {
                    length: octant.data.len() as u32,
                    points: pool.next(data).unwrap(),
                }
            })
            .collect::<Vec<OctantBuffer<512>>>();

        // let indirect_args_pool = CpuBufferPool::new(device.clone(), BufferUsage::all());

        println!("num octants: {:}", octants.len());
        OctreeRenderer {
            device,
            queue,
            pipeline,
            set,
            uniforms,
            vs,
            fs,
            octree,
            octants,
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

    pub fn frustum_culling(&mut self, camera: &PerspectiveCamera) {}

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.pipeline = Self::build_pipeline(
            self.vs.clone(),
            self.fs.clone(),
            self.pipeline.subpass().clone(),
            viewport,
            self.pipeline.device().clone(),
        );
    }

    pub fn render(&self) -> Arc<SecondaryAutoCommandBuffer> {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(
                0,
                self.uniforms.pool_chunk().clone(),
            )],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
            self.pipeline.subpass().clone(),
        )
        .unwrap();

        builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set.clone(),
            );

        for octant in self.octants.iter() {
            builder
                .bind_vertex_buffers(0, octant.points.clone())
                .draw(octant.length, 1, 0, 0)
                .unwrap();
        }
        Arc::new(builder.build().unwrap())
    }

    pub fn set_point_size(&mut self, point_size: u32) {
        self.uniforms.update(vs::UniformData {
            point_size,
            ..*self.uniforms.data()
        });
    }

    pub fn set_camera(&mut self, camera: &Camera<impl Projection>) {
        self.uniforms.update(vs::UniformData {
            view: camera.view().clone().into(),
            proj: camera.projection().clone().into(),
            znear: *camera.znear(),
            zfar: *camera.zfar(),
            ..*self.uniforms.data()
        });
    }
}
