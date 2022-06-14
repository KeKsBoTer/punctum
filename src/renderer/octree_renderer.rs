use crate::{
    camera::{Camera, Projection},
    vertex::Vertex,
    Octree, Viewport,
};
use nalgebra::{matrix, Vector3};
use std::sync::{Arc, RwLock};
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

pub struct OctantBuffer<const T: usize> {
    length: u32,
    points: Arc<CpuBufferPoolSubbuffer<[Vertex<f32, f32>; T], Arc<StdMemoryPool>>>,
}

pub struct OctreeRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,

    pipeline: RwLock<Arc<GraphicsPipeline>>,

    uniforms: RwLock<UniformBuffer<vs::UniformData>>,
    fs: Arc<ShaderModule>,
    vs: Arc<ShaderModule>,

    octree: Arc<Octree<f32, f32>>,
    octants: RwLock<Vec<OctantBuffer<8192>>>,
    octants_pool: CpuBufferPool<[Vertex<f32, f32>; 8192]>,
}

impl OctreeRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass,
        viewport: Viewport,
        octree: Arc<Octree<f32, f32>>,
    ) -> Self {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pipeline =
            Self::build_pipeline(vs.clone(), fs.clone(), subpass, viewport, device.clone());

        let max_size = octree.size();
        let center = octree.center();

        let scale_size = 100.;
        let scale = scale_size / max_size;
        let world = matrix![
            scale, 0., 0., -scale * center.x;
            0., scale, 0., -scale * center.y;
            0., 0., scale, -scale * center.z;
            0., 0., 0.   , 1.
        ];

        let uniforms = RwLock::new(UniformBuffer::new(
            device.clone(),
            Some(vs::UniformData {
                world: world.into(),
                ..vs::UniformData::default()
            }),
        ));

        let pool: CpuBufferPool<[Vertex<f32, f32>; 8192]> =
            CpuBufferPool::new(device.clone(), BufferUsage::vertex_buffer());

        println!("center: {:?}, size: {}", center, scale_size / max_size);

        OctreeRenderer {
            device,
            queue,
            pipeline: RwLock::new(pipeline),
            uniforms,
            vs,
            fs,
            octree,
            octants: RwLock::new(Vec::new()),
            octants_pool: pool,
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

    pub fn frustum_culling(&self) {
        let u = {
            let uniforms = self.uniforms.read().unwrap();
            uniforms.data().clone()
        };
        let view_transform = u.proj * u.view * u.world;

        let visible = self
            .octree
            .into_iter()
            .filter(|octant| {
                let size = octant.size / 2.;
                let points = [
                    octant.center - Vector3::new(-size, -size, -size),
                    octant.center - Vector3::new(size, -size, -size),
                    octant.center - Vector3::new(-size, size, -size),
                    octant.center - Vector3::new(size, size, -size),
                    octant.center - Vector3::new(-size, -size, size),
                    octant.center - Vector3::new(size, -size, size),
                    octant.center - Vector3::new(-size, size, size),
                    octant.center - Vector3::new(size, size, size),
                ];

                points
                    .map(|p| {
                        let screen_space = view_transform * p.to_homogeneous();
                        let n_pos = screen_space.xyz() / screen_space.w;
                        n_pos.abs() <= Vector3::new(1., 1., 1.)
                    })
                    .contains(&true)
            })
            .map(|octant| {
                let mut data = [Vertex::<f32, f32>::default(); 8192];
                for (i, p) in octant.data.iter().enumerate() {
                    data[i] = Vertex {
                        position: p.position.into(),
                        color: p.color,
                    };
                }
                OctantBuffer {
                    length: octant.data.len() as u32,
                    points: self.octants_pool.next(data).unwrap(),
                }
            })
            .collect::<Vec<OctantBuffer<8192>>>();

        println!("visible octants: {}", visible.len());
        let mut octants = self.octants.write().unwrap();
        *octants = visible;
    }

    pub fn set_viewport(&self, viewport: Viewport) {
        let mut pipeline = self.pipeline.write().unwrap();
        *pipeline = Self::build_pipeline(
            self.vs.clone(),
            self.fs.clone(),
            pipeline.subpass().clone(),
            viewport,
            pipeline.device().clone(),
        );
    }

    pub fn render(&self) -> Arc<SecondaryAutoCommandBuffer> {
        let pipeline = self.pipeline.read().unwrap();

        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let uniform_buffer = {
            let uniforms = self.uniforms.read().unwrap();
            uniforms.pool_chunk().clone()
        };
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
            pipeline.subpass().clone(),
        )
        .unwrap();

        builder
            .bind_pipeline_graphics(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                set.clone(),
            );

        for octant in self.octants.read().unwrap().iter() {
            builder
                .bind_vertex_buffers(0, octant.points.clone())
                .draw(octant.length, 1, 0, 0)
                .unwrap();
        }
        Arc::new(builder.build().unwrap())
    }

    pub fn set_point_size(&self, point_size: u32) {
        let mut uniforms = self.uniforms.write().unwrap();
        let current = *uniforms.data();
        uniforms.update(vs::UniformData {
            point_size,
            ..current
        });
    }

    pub fn set_camera(&self, camera: &Camera<impl Projection>) {
        let mut uniforms = self.uniforms.write().unwrap();
        let current = *uniforms.data();
        uniforms.update(vs::UniformData {
            view: camera.view().clone().into(),
            proj: camera.projection().clone().into(),
            znear: *camera.znear(),
            zfar: *camera.zfar(),
            ..current
        });
    }
}
