use crate::{
    camera::{Camera, Projection},
    octree::OctreeIter,
    sh::calc_sh_grid,
    vertex::Vertex,
    Octree, SHVertex, Viewport,
};
use nalgebra::matrix;
use std::{
    collections::HashMap,
    num::NonZeroU64,
    sync::{Arc, Mutex, RwLock},
};
use vulkano::{
    buffer::{
        cpu_pool::CpuBufferPoolSubbuffer, BufferAccess, BufferDeviceAddressError, BufferUsage,
        CpuAccessibleBuffer, CpuBufferPool,
    },
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::{
        pool::standard::StdDescriptorPoolAlloc, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage},
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
    sampler::{Sampler, SamplerCreateInfo},
    shader::ShaderModule,
    sync::{self, GpuFuture},
    VulkanObject,
};

use super::UniformBuffer;

mod vs {
    use bytemuck::{Pod, Zeroable};
    use nalgebra::{Matrix4, Point3};

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/pointcloud_sh.vert",

        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy,Pod, Zeroable)]
        }
    }

    #[derive(Clone, Copy, Zeroable)]
    #[repr(C)]
    pub struct UniformData {
        pub world: Matrix4<f32>,
        pub view: Matrix4<f32>,
        pub proj: Matrix4<f32>,
        pub camera_pos: Point3<f32>,
        pub point_size: u32,
        pub znear: f32,
        pub zfar: f32,
    }

    unsafe impl Pod for UniformData {}

    impl Default for UniformData {
        fn default() -> Self {
            Self {
                world: Matrix4::identity().into(),
                view: Matrix4::identity().into(),
                proj: Matrix4::identity().into(),
                camera_pos: Point3::origin(),
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

#[derive(Clone)]
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
    loaded_octants: RwLock<HashMap<u64, OctantBuffer<8192>>>,
    octants_pool: CpuBufferPool<[Vertex<f32, f32>; 8192]>,
    // reference_buffer: Arc<CpuAccessibleBuffer<vs::ty::ObjDesc>>,

    // vertex_buffer: Arc<CpuAccessibleBuffer<i32>>,
    // phantom_buffer: Arc<DeviceLocalBuffer<[Vertex<f32, f32>]>>,

    // indirect_draw_cmd: Arc<CpuBufferPoolChunk<DrawIndirectCommand, Arc<StdMemoryPool>>>,
    sh_vertex_buffer: Arc<CpuAccessibleBuffer<[SHVertex<f32, 121>]>>,
    sh_vertex_buffer_len: RwLock<u32>,
    sh_set: Arc<PersistentDescriptorSet<StdDescriptorPoolAlloc>>,
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

        // let reference_buffer =
        //     CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::uniform_buffer(), false);

        let uniforms = RwLock::new(UniformBuffer::new(
            device.clone(),
            Some(vs::UniformData {
                world: world.into(),
                ..vs::UniformData::default()
            }),
        ));

        // let indirect_args_pool: CpuBufferPool<DrawIndirectCommand> =
        //     CpuBufferPool::new(device.clone(), BufferUsage::indirect_buffer());
        // let indirect_commands = [DrawIndirectCommand {
        //     vertex_count: 10,
        //     instance_count: 1,
        //     first_vertex: 0,
        //     first_instance: 0,
        // }];
        // let indirect_draw_cmd = indirect_args_pool.chunk(indirect_commands).unwrap();

        // let vertex_buffer = CpuAccessibleBuffer::from_data(
        //     device.clone(),
        //     BufferUsage {
        //         device_address: true,
        //         storage_buffer: true,
        //         vertex_buffer: true,
        //         ..BufferUsage::none()
        //     },
        //     false,
        //     -7,
        // )
        // .unwrap();

        // let phantom_buffer: Arc<DeviceLocalBuffer<[Vertex<f32, f32>]>> =
        //     DeviceLocalBuffer::array(device.clone(), 1, BufferUsage::all(), [queue.family()])
        //         .unwrap();

        // let addr = raw_device_address(&phantom_buffer).unwrap();

        // let reference_buffer = CpuAccessibleBuffer::from_data(
        //     device.clone(),
        //     BufferUsage {
        //         uniform_buffer: true,
        //         storage_buffer: true,
        //         ..BufferUsage::default()
        //     },
        //     false,
        //     vs::ty::ObjDesc {
        //         vertexAddress: addr.into(),
        //     },
        // )
        // .unwrap();

        let pool: CpuBufferPool<[Vertex<f32, f32>; 8192]> =
            CpuBufferPool::new(device.clone(), BufferUsage::vertex_buffer());

        let (sh_images, sh_sampler) = calc_sh_images(device.clone(), queue.clone(), 128, 10);

        let layout = pipeline.layout().set_layouts().get(1).unwrap();

        let empty_sh_vertices = (0..octree.num_octants())
            .into_iter()
            .map(|_| SHVertex::default())
            .collect::<Vec<SHVertex<f32, 121>>>();
        let num_empty_sh_vertices = empty_sh_vertices.len() as u32;
        let sh_vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                vertex_buffer: true,
                storage_buffer: true,
                ..BufferUsage::default()
            },
            false,
            empty_sh_vertices,
        )
        .unwrap();

        let sh_set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, sh_images.clone(), sh_sampler.clone()),
                WriteDescriptorSet::buffer(1, sh_vertex_buffer.clone()),
            ],
        )
        .unwrap();

        OctreeRenderer {
            device,
            queue,
            pipeline: RwLock::new(pipeline),
            uniforms,
            vs,
            fs,
            octree,
            loaded_octants: RwLock::new(HashMap::new()),
            octants_pool: pool,
            // reference_buffer,
            // vertex_buffer,
            // phantom_buffer,
            // indirect_draw_cmd,
            sh_vertex_buffer: sh_vertex_buffer,
            sh_vertex_buffer_len: RwLock::new(num_empty_sh_vertices),
            sh_set,
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
            .vertex_input_state(BuffersDefinition::new().vertex::<SHVertex<f32, 121>>())
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

        let octants = self.loaded_octants.read().unwrap();

        let mut sh_vertices = Vec::new();

        let mut new = 0;
        let mut new_octants = HashMap::new();

        for OctreeIter { octant, bbox } in self.octree.into_octant_iterator() {
            if bbox.at_least_one_point_visible(&view_transform) {
                let id = octant.id();
                if let Some(o) = octants.get(&id) {
                    new_octants.insert(id, o.clone());
                } else {
                    let mut data = [Vertex::<f32, f32>::default(); 8192];
                    for (i, p) in octant.points().iter().enumerate() {
                        data[i] = Vertex {
                            position: p.position.into(),
                            color: p.color,
                        };
                    }
                    new += 1;
                    new_octants.insert(
                        id,
                        OctantBuffer {
                            length: octant.points().len() as u32,
                            points: self.octants_pool.next(data).unwrap(),
                        },
                    );

                    if let Some(sh) = octant.sh_approximation {
                        sh_vertices.push(sh);
                    }
                }
            }
        }
        if new > 0 {
            println!("new octants: {}", new);
        }
        drop(octants);

        let mut octants = self.loaded_octants.write().unwrap();
        *octants = new_octants;

        {
            let mut vertex_buffer = self.sh_vertex_buffer.write().unwrap();
            for (i, sh) in sh_vertices.iter().enumerate() {
                vertex_buffer[i] = *sh;
            }
            let mut size = self.sh_vertex_buffer_len.write().unwrap();
            *size = sh_vertices.len() as u32;
            println!("update!");
        }
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

        // TODO maybe pass chunks as buffer_array ?
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
                [set.clone(), self.sh_set.clone()].to_vec(),
            );

        // for (_, octant) in self.loaded_octants.read().unwrap().iter() {
        //     builder
        //         .bind_vertex_buffers(0, octant.points.clone())
        //         .draw(octant.length, 1, 0, 0)
        //         .unwrap();
        // }

        // builder
        //     .bind_vertex_buffers(0, self.phantom_buffer.clone())
        //     .draw_indirect(self.indirect_draw_cmd.clone())
        //     .unwrap();

        builder
            .bind_vertex_buffers(0, self.sh_vertex_buffer.clone())
            .draw(*self.sh_vertex_buffer_len.read().unwrap(), 1, 0, 0)
            .unwrap();

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
            camera_pos: camera.position().clone(),
            znear: *camera.znear(),
            zfar: *camera.zfar(),
            ..current
        });
    }
}

fn raw_device_address(buffer: impl BufferAccess) -> Result<NonZeroU64, BufferDeviceAddressError> {
    let inner = buffer.inner();
    let device = buffer.device();

    // VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324
    if !device.enabled_features().buffer_device_address {
        return Err(BufferDeviceAddressError::FeatureNotEnabled);
    }

    // VUID-VkBufferDeviceAddressInfo-buffer-02601
    if !inner.buffer.usage().device_address {
        return Err(BufferDeviceAddressError::BufferMissingUsage);
    }

    unsafe {
        let info = ash::vk::BufferDeviceAddressInfo {
            buffer: inner.buffer.internal_object(),
            ..Default::default()
        };
        let ptr = device
            .fns()
            .ext_buffer_device_address
            .get_buffer_device_address_ext(device.internal_object(), &info);

        if ptr == 0 {
            panic!("got null ptr from a valid GetBufferDeviceAddressEXT call");
        }

        Ok(NonZeroU64::new_unchecked(ptr + inner.offset))
    }
}

fn calc_sh_images(
    device: Arc<Device>,
    queue: Arc<Queue>,
    img_size: u32,
    lmax: u64,
) -> (Arc<ImageView<ImmutableImage>>, Arc<Sampler>) {
    let images = calc_sh_grid(lmax, img_size);
    let dimensions = ImageDimensions::Dim2d {
        width: img_size,
        height: img_size,
        array_layers: images.len() as u32,
    };

    let (sh_images, future) = ImmutableImage::from_iter(
        images.into_iter().flatten().collect::<Vec<f32>>(),
        dimensions,
        vulkano::image::MipmapsCount::One,
        Format::R32_SFLOAT,
        queue.clone(),
    )
    .unwrap();

    sync::now(device.clone())
        .join(future)
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let sh_images_view = ImageView::new_default(sh_images.clone()).unwrap();

    let sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();

    return (sh_images_view, sampler);
}
