use crate::{
    camera::{Camera, Projection, ViewFrustum},
    octree::OctreeIter,
    sh::calc_sh_grid,
    vertex::Vertex,
    CubeBoundingBox, Octree, SHVertex, Viewport,
};
use nalgebra::{distance, Matrix4};
use std::{
    collections::HashMap,
    f32::consts::PI,
    sync::{Arc, RwLock},
};
use vulkano::{
    buffer::{
        cpu_pool::CpuBufferPoolSubbuffer, BufferAccess, BufferUsage, CpuAccessibleBuffer,
        CpuBufferPool,
    },
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
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
};

use super::UniformBuffer;

#[derive(Clone)]
pub struct OctantBuffer<const T: usize> {
    length: u32,
    points: Arc<CpuBufferPoolSubbuffer<[Vertex<f32, f32>; T], Arc<StdMemoryPool>>>,
}

pub struct OctreeRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,

    uniforms: RwLock<UniformBuffer<vs::ty::UniformData>>,
    frustum: RwLock<ViewFrustum<f32>>,

    octree: Arc<Octree<f32, f32>>,

    subpass: Subpass,
    sh_renderer: SHRenderer,
    octant_renderer: OctantRenderer,
    screen_height: RwLock<u32>,
}

impl OctreeRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass,
        viewport: Viewport,
        octree: Arc<Octree<f32, f32>>,
        camera: &Camera<impl Projection>,
    ) -> Self {
        let world = Matrix4::identity();
        let viewport_height = viewport.size()[1] as u32;

        let uniforms = RwLock::new(UniformBuffer::new(
            device.clone(),
            Some(vs::ty::UniformData {
                view: camera.view().clone().into(),
                proj: camera.projection().clone().into(),
                world: world.into(),
                camera_pos: camera.position().clone().into(),
                ..vs::ty::UniformData::default()
            }),
        ));
        let frustum = RwLock::new(camera.extract_planes_from_projmat(true));

        let sh_renderer = SHRenderer::new(
            device.clone(),
            queue.clone(),
            subpass.clone(),
            viewport.clone(),
        );
        let octant_renderer =
            OctantRenderer::new(device.clone(), queue.clone(), subpass.clone(), viewport);

        Self {
            device,
            queue,
            uniforms,
            frustum,
            octree,
            subpass,
            sh_renderer,
            octant_renderer,
            screen_height: RwLock::new(viewport_height),
        }
    }

    pub fn frustum_culling(&self) {
        let u = {
            let uniforms = self.uniforms.read().unwrap();
            uniforms.data.clone()
        };

        let frustum = self.frustum.read().unwrap();
        let visible_octants = self.octree.visible_octants(&frustum);

        let camera_fov = PI / 2.;
        let screen_height = self.screen_height.read().unwrap().clone();
        let threshold_fn = |bbox: &CubeBoundingBox<f32>| {
            let d = distance(&bbox.center, &u.camera_pos.into());
            let radius = bbox.outer_radius();
            let a = 2. * (radius / d).atan();
            return a / camera_fov <= 2. / screen_height as f32;
        };

        self.sh_renderer
            .frustum_culling(&visible_octants, threshold_fn);
        self.octant_renderer
            .frustum_culling(&visible_octants, threshold_fn);
    }

    pub fn set_viewport(&self, viewport: Viewport) {
        let viewport_height = viewport.size()[1] as u32;
        self.sh_renderer.set_viewport(viewport.clone());
        self.octant_renderer.set_viewport(viewport);

        let mut screen_height = self.screen_height.write().unwrap();
        *screen_height = viewport_height;
    }

    pub fn render(
        &self,
        render_octants: bool,
        render_shs: bool,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let uniform_buffer = {
            let uniforms = self.uniforms.read().unwrap();
            uniforms.buffer().clone()
        };

        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
            self.subpass.clone(),
        )
        .unwrap();

        if render_shs {
            self.sh_renderer
                .render(uniform_buffer.clone(), &mut builder);
        }
        if render_octants {
            self.octant_renderer.render(uniform_buffer, &mut builder);
        }

        Arc::new(builder.build().unwrap())
    }

    pub fn set_point_size(&self, point_size: u32) {
        let mut uniforms = self.uniforms.write().unwrap();
        uniforms.data.point_size = point_size;
    }

    pub fn set_highlight_sh(&self, highlight_sh: bool) {
        let mut uniforms = self.uniforms.write().unwrap();
        uniforms.data.highlight_sh = highlight_sh as u32;
    }

    pub fn update_uniforms(&self) {
        let mut uniforms = self.uniforms.write().unwrap();
        uniforms.update_buffer();
    }

    pub fn set_camera(&self, camera: &Camera<impl Projection>) {
        let mut uniforms = self.uniforms.write().unwrap();
        uniforms.data.view = camera.view().clone().into();
        uniforms.data.proj = camera.projection().clone().into();
        uniforms.data.camera_pos = camera.position().clone().into();

        let mut frustum = self.frustum.write().unwrap();
        *frustum = camera.extract_planes_from_projmat(true);
    }
}

// fn raw_device_address(buffer: impl BufferAccess) -> Result<NonZeroU64, BufferDeviceAddressError> {
//     let inner = buffer.inner();
//     let device = buffer.device();

//     // VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324
//     if !device.enabled_features().buffer_device_address {
//         return Err(BufferDeviceAddressError::FeatureNotEnabled);
//     }

//     // VUID-VkBufferDeviceAddressInfo-buffer-02601
//     if !inner.buffer.usage().device_address {
//         return Err(BufferDeviceAddressError::BufferMissingUsage);
//     }

//     unsafe {
//         let info = ash::vk::BufferDeviceAddressInfo {
//             buffer: inner.buffer.internal_object(),
//             ..Default::default()
//         };
//         let ptr = device
//             .fns()
//             .ext_buffer_device_address
//             .get_buffer_device_address_ext(device.internal_object(), &info);

//         if ptr == 0 {
//             panic!("got null ptr from a valid GetBufferDeviceAddressEXT call");
//         }

//         Ok(NonZeroU64::new_unchecked(ptr + inner.offset))
//     }
// }

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/pointcloud.frag"
    }
}

mod vs_sh {

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/pointcloud_sh.vert",

        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy,Pod, Zeroable)]
        }
    }
}

struct SHRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,

    pipeline: RwLock<Arc<GraphicsPipeline>>,

    fs: Arc<ShaderModule>,
    vs: Arc<ShaderModule>,

    sh_vertex_buffer: RwLock<Arc<CpuAccessibleBuffer<[SHVertex<f32, 121>]>>>,
    sh_vertex_buffer_len: RwLock<u32>,

    sh_images: Arc<ImageView<ImmutableImage>>,
    sh_sampler: Arc<Sampler>,
}

impl SHRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass,
        viewport: Viewport,
    ) -> Self {
        let vs = vs_sh::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pipeline = build_pipeline::<SHVertex<f32, 121>>(
            vs.clone(),
            fs.clone(),
            subpass,
            viewport,
            device.clone(),
        );

        let (sh_images, sh_sampler) = Self::calc_sh_images(device.clone(), queue.clone(), 128, 10);

        let num_empty_sh_vertices = 1;
        let sh_vertex_buffer = RwLock::new(
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage {
                    vertex_buffer: true,
                    storage_buffer: true,
                    ..BufferUsage::default()
                },
                false,
                (0..1).map(|_| SHVertex::<f32, 121>::default()),
            )
            .unwrap(),
        );

        Self {
            device,
            queue,
            pipeline: RwLock::new(pipeline),
            vs,
            fs,
            sh_vertex_buffer: sh_vertex_buffer,
            sh_vertex_buffer_len: RwLock::new(num_empty_sh_vertices),
            sh_images,
            sh_sampler,
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

        let sh_vertex_buffer = self.sh_vertex_buffer.read().unwrap();

        let sh_set = PersistentDescriptorSet::new(
            set_layouts.get(1).unwrap().clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    self.sh_images.clone(),
                    self.sh_sampler.clone(),
                ),
                WriteDescriptorSet::buffer(1, sh_vertex_buffer.clone()),
            ],
        )
        .unwrap();

        builder
            .bind_pipeline_graphics(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                [set.clone(), sh_set.clone()].to_vec(),
            );

        builder
            .bind_vertex_buffers(0, sh_vertex_buffer.clone())
            .draw(*self.sh_vertex_buffer_len.read().unwrap(), 1, 0, 0)
            .unwrap();
    }

    pub fn frustum_culling<'a, F>(
        &self,
        visible_octants: &Vec<OctreeIter<'a, f32, f32>>,
        threshold_fn: F,
    ) where
        F: Fn(&CubeBoundingBox<f32>) -> bool,
    {
        let mut sh_vertices = Vec::new();

        for OctreeIter { octant, bbox } in visible_octants {
            if let Some(sh) = octant.sh_approximation {
                if threshold_fn(bbox) {
                    sh_vertices.push(sh);
                }
            }
        }

        // TODO write to buffer instead of creating a new one
        // maybe use CPU pool?
        let new_len = sh_vertices.len() as u32;
        if new_len > 0 {
            let mut vertex_buffer = self.sh_vertex_buffer.write().unwrap();
            *vertex_buffer = CpuAccessibleBuffer::from_iter(
                self.device.clone(),
                BufferUsage {
                    vertex_buffer: true,
                    storage_buffer: true,
                    ..BufferUsage::default()
                },
                false,
                sh_vertices,
            )
            .unwrap();
        }
        let mut size = self.sh_vertex_buffer_len.write().unwrap();
        *size = new_len;
    }

    pub fn set_viewport(&self, viewport: Viewport) {
        let mut pipeline = self.pipeline.write().unwrap();
        *pipeline = build_pipeline::<SHVertex<f32, 121>>(
            self.vs.clone(),
            self.fs.clone(),
            pipeline.subpass().clone(),
            viewport,
            pipeline.device().clone(),
        );
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

        let sampler =
            Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();

        return (sh_images_view, sampler);
    }
}

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

struct OctantRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,

    pipeline: RwLock<Arc<GraphicsPipeline>>,

    fs: Arc<ShaderModule>,
    vs: Arc<ShaderModule>,

    loaded_octants: RwLock<HashMap<u64, OctantBuffer<8192>>>,
    octants_pool: CpuBufferPool<[Vertex<f32, f32>; 8192]>,
}

impl OctantRenderer {
    fn new(device: Arc<Device>, queue: Arc<Queue>, subpass: Subpass, viewport: Viewport) -> Self {
        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pipeline = build_pipeline::<Vertex<f32, f32>>(
            vs.clone(),
            fs.clone(),
            subpass,
            viewport,
            device.clone(),
        );

        let pool: CpuBufferPool<[Vertex<f32, f32>; 8192]> =
            CpuBufferPool::new(device.clone(), BufferUsage::vertex_buffer());

        Self {
            device,
            queue,
            pipeline: RwLock::new(pipeline),
            vs,
            fs,
            loaded_octants: RwLock::new(HashMap::new()),
            octants_pool: pool,
        }
    }
    fn render(
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

        for (_, octant) in self.loaded_octants.read().unwrap().iter() {
            builder
                .bind_vertex_buffers(0, octant.points.clone())
                .draw(octant.length, 1, 0, 0)
                .unwrap();
        }
    }

    pub fn frustum_culling<'a, F>(
        &self,
        visible_octants: &Vec<OctreeIter<'a, f32, f32>>,
        threshold_fn: F,
    ) where
        F: Fn(&CubeBoundingBox<f32>) -> bool,
    {
        let octants = self.loaded_octants.read().unwrap();
        let mut new_octants = HashMap::new();

        for OctreeIter { octant, bbox } in visible_octants {
            // check if octant is larger than one pixel on screen
            if !threshold_fn(bbox) {
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
                    new_octants.insert(
                        id,
                        OctantBuffer {
                            length: octant.points().len() as u32,
                            points: self.octants_pool.next(data).unwrap(),
                        },
                    );
                }
            }
        }
        drop(octants);

        let mut octants = self.loaded_octants.write().unwrap();
        *octants = new_octants;
    }

    fn set_viewport(&self, viewport: Viewport) {
        let mut pipeline = self.pipeline.write().unwrap();
        *pipeline = build_pipeline::<Vertex<f32, f32>>(
            self.vs.clone(),
            self.fs.clone(),
            pipeline.subpass().clone(),
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
