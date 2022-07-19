use crate::{
    camera::{Camera, Projection, ViewFrustum},
    octree::OctreeIter,
    sh::calc_sh_grid,
    vertex::{IndexVertex, Vertex},
    CubeBoundingBox, Octree, Viewport,
};
use nalgebra::{distance, Matrix4, Point3};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::HashMap,
    f64::consts::PI,
    sync::{Arc, RwLock},
};
use vulkano::{
    buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage},
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

pub struct OctreeRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,

    uniforms: RwLock<UniformBuffer<vs::ty::UniformData>>,
    frustum: RwLock<ViewFrustum<f32>>,

    octree: Arc<Octree<f32, f32>>,

    subpass: Subpass,
    octant_renderer: OctantRenderer,
    sh_renderer: Option<SHRenderer>,
    screen_height: RwLock<u32>,
}

#[derive(Debug, Clone, Copy)]
pub enum CullingMode {
    Mixed,
    SHOnly,
    OctantsOnly,
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

        let sh_renderer = if octree.into_iter().any(|o| o.sh_approximation.is_some()) {
            Some(SHRenderer::new(
                device.clone(),
                queue.clone(),
                subpass.clone(),
                viewport.clone(),
                octree.clone(),
            ))
        } else {
            None
        };
        let octant_renderer =
            OctantRenderer::new(device.clone(), subpass.clone(), viewport, octree.clone());

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

    pub fn frustum_culling(&self, mode: CullingMode) {
        let u = {
            let uniforms = self.uniforms.read().unwrap();
            uniforms.data.clone()
        };

        let frustum = self.frustum.read().unwrap();

        let camera_fov = PI / 2.;
        let screen_height = self.screen_height.read().unwrap().clone();
        let cam_pos: Point3<f32> = u.camera_pos.into();
        let threshold_fn = |bbox: &CubeBoundingBox<f64>| {
            let d = distance(&bbox.center, &cam_pos.cast());
            let radius = bbox.outer_radius();
            let a = 2. * (radius / d).atan();
            return a / camera_fov <= 1. / screen_height as f64;
        };

        // divide octants into the ones that are fully rendered and the
        // ones that get rendered with the spherical harmonics representation
        let (visible_sh, visible_octants): (Vec<_>, Vec<_>) = self
            .octree
            .visible_octants(&frustum)
            .into_par_iter()
            .partition(|o| match mode {
                CullingMode::Mixed => o.octant.sh_approximation.is_some() && threshold_fn(&o.bbox),
                CullingMode::SHOnly => true,
                CullingMode::OctantsOnly => false,
            });

        if let Some(sh_renderer) = &self.sh_renderer {
            sh_renderer.frustum_culling(&visible_sh);
        }
        self.octant_renderer.frustum_culling(&visible_octants);
    }

    pub fn set_viewport(&self, viewport: Viewport) {
        let viewport_height = viewport.size()[1] as u32;
        if let Some(sh_renderer) = &self.sh_renderer {
            sh_renderer.set_viewport(viewport.clone());
        }
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
            if let Some(sh_renderer) = &self.sh_renderer {
                sh_renderer.render(uniform_buffer.clone(), &mut builder);
            }
        }
        if render_octants {
            self.octant_renderer
                .render(uniform_buffer.clone(), &mut builder);
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

    pipeline: RwLock<Arc<GraphicsPipeline>>,

    fs: Arc<ShaderModule>,
    vs: Arc<ShaderModule>,

    sh_index_buffer: RwLock<Option<Arc<CpuAccessibleBuffer<[IndexVertex]>>>>,
    sh_mapping: HashMap<u64, u32>,

    sh_set: Arc<PersistentDescriptorSet>,
}

impl SHRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        subpass: Subpass,
        viewport: Viewport,
        octree: Arc<Octree<f32, f32>>,
    ) -> Self {
        let vs = vs_sh::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");

        let pipeline = build_pipeline::<IndexVertex>(
            vs.clone(),
            fs.clone(),
            subpass,
            viewport,
            device.clone(),
        );

        let (sh_images, sh_sampler) = Self::calc_sh_images(device.clone(), queue.clone(), 128, 10);

        let num_octants = octree.num_octants() as usize;
        let mut shs = Vec::with_capacity(num_octants);
        let mut sh_mapping = HashMap::with_capacity(num_octants);
        for (i, octant) in octree
            .into_iter()
            .filter(|o| o.sh_approximation.is_some())
            .enumerate()
        {
            shs.push(octant.sh_approximation.unwrap());
            sh_mapping.insert(octant.id(), i as u32);
        }

        let sh_vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                storage_buffer: true,
                ..BufferUsage::default()
            },
            false,
            shs,
        )
        .unwrap();

        let set_layouts = pipeline.layout().set_layouts();
        let sh_set = PersistentDescriptorSet::new(
            set_layouts.get(1).unwrap().clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, sh_images.clone(), sh_sampler.clone()),
                WriteDescriptorSet::buffer(1, sh_vertex_buffer.clone()),
            ],
        )
        .unwrap();

        Self {
            device,
            pipeline: RwLock::new(pipeline),
            vs,
            fs,
            sh_index_buffer: RwLock::new(None),
            sh_mapping,
            sh_set,
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
                [set.clone(), self.sh_set.clone()].to_vec(),
            );

        let vertex_indices = self.sh_index_buffer.read().unwrap();
        if vertex_indices.is_some() {
            let indices = vertex_indices.as_ref().unwrap();
            builder
                .bind_vertex_buffers(0, indices.clone())
                .draw(indices.into_buffer_slice().len() as u32, 1, 0, 0)
                .unwrap();
        }
    }

    pub fn frustum_culling<'a>(&self, visible_octants: &Vec<OctreeIter<'a, f32, f32>>) {
        // calculate indices to all visible sh vertices
        let mut sh_vertices: Vec<IndexVertex> = visible_octants
            .par_iter()
            .filter_map(|octant| {
                if octant.octant.sh_approximation.is_some() {
                    Some(IndexVertex {
                        index: *self.sh_mapping.get(&octant.octant.id()).unwrap(),
                    })
                } else {
                    None
                }
            })
            .collect();

        sh_vertices.sort_by_key(|v| v.index);

        let new_len = sh_vertices.len() as u32;
        if new_len > 0 {
            let new_buffer = CpuAccessibleBuffer::from_iter(
                self.device.clone(),
                BufferUsage {
                    vertex_buffer: true,
                    ..BufferUsage::default()
                },
                false,
                sh_vertices,
            )
            .unwrap();
            let mut sh_index_buffer = self.sh_index_buffer.write().unwrap();
            *sh_index_buffer = Some(new_buffer);
        }
    }

    pub fn set_viewport(&self, viewport: Viewport) {
        let mut pipeline = self.pipeline.write().unwrap();
        *pipeline = build_pipeline::<IndexVertex>(
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
    pipeline: RwLock<Arc<GraphicsPipeline>>,

    fs: Arc<ShaderModule>,
    vs: Arc<ShaderModule>,

    /// list of visible octants
    visible_octants: RwLock<Vec<(u32, u32)>>,

    /// all octree vertices in one block of memory
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex<f32, f32>]>>,
    /// maps octant ids to (location,size) in vertex_buffer
    vertex_mapping: HashMap<u64, (u32, u32)>,
}

impl OctantRenderer {
    fn new(
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
            subpass,
            viewport,
            device.clone(),
        );

        // upload all vertices to GPU into one large buffer
        // stores memory mapping in hash map for later access
        let vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex<f32, f32>]>> = unsafe {
            CpuAccessibleBuffer::uninitialized_array(
                device.clone(),
                octree.num_points(),
                BufferUsage::vertex_buffer(),
                false,
            )
            .unwrap()
        };
        let mut offset = 0;
        let mut vertex_mapping: HashMap<u64, (u32, u32)> =
            HashMap::with_capacity(octree.num_octants() as usize);
        {
            let mut vertices = vertex_buffer.write().unwrap();
            for octant in octree.into_iter() {
                let octant_size = octant.points().len();
                vertex_mapping.insert(octant.id(), (offset, octant_size as u32));
                for (i, p) in octant.points().iter().enumerate() {
                    vertices[offset as usize + i] = *p;
                }
                offset += octant_size as u32;
            }
        }

        Self {
            pipeline: RwLock::new(pipeline),
            vs,
            fs,
            visible_octants: RwLock::new(Vec::new()),

            vertex_buffer,
            vertex_mapping,
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

        let visible_octants = self.visible_octants.read().unwrap();
        builder.bind_vertex_buffers(0, self.vertex_buffer.clone());
        let mut last_offset = 0;
        let mut last_len = 0;
        for (offset, len) in visible_octants.iter() {
            if last_offset + last_len == *offset as u32 {
                last_len += *len as u32;
            } else {
                if last_len > 0 {
                    builder.draw(last_len, 1, last_offset, 0).unwrap();
                }
                last_offset = *offset as u32;
                last_len = *len as u32;
            }
        }
        builder.draw(last_len, 1, last_offset, 0).unwrap();
    }

    pub fn frustum_culling<'a>(&self, visible_octants: &Vec<OctreeIter<'a, f32, f32>>) {
        let mut new_octants: Vec<(u32, u32)> = visible_octants
            .par_iter()
            .map(|oc| {
                let id = oc.octant.id();
                self.vertex_mapping.get(&id).unwrap().clone()
            })
            .collect();

        // merge draw calls together that overlap
        // e.g. (0,17) and (17,5) would be merged to (0,23)
        new_octants.sort_by_key(|(offset, _)| *offset);
        let mut merged = Vec::new();
        let mut last_offset = 0;
        let mut last_len = 0;
        for (offset, len) in new_octants.iter() {
            if last_offset + last_len == *offset as u32 {
                last_len += *len as u32;
            } else {
                if last_len > 0 {
                    merged.push((last_offset, last_len));
                }
                last_offset = *offset as u32;
                last_len = *len as u32;
            }
        }
        if last_len > 0 {
            merged.push((last_offset, last_len));
        }

        {
            let mut octants = self.visible_octants.write().unwrap();
            *octants = merged;
        }
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
