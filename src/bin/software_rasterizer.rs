use nalgebra::{Matrix4, Point3, Vector3};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use structopt::StructOpt;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Features;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::StorageImage;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;

use pbr::ProgressBar;
use vulkano::device::DeviceExtensions;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano_win::VkSurfaceBuild;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event_loop::EventLoop;
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::WindowBuilder;

use winit::event::{ElementState, Event, KeyboardInput, MouseButton, WindowEvent};
use winit::event_loop::ControlFlow;

use punctum::{
    get_render_pass, select_physical_device, CameraController, Octree, PerspectiveCamera,
    SurfaceFrame, TeeReader, Vertex, Viewport,
};

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        types_meta: {
            use bytemuck::{Pod, Zeroable};
            #[derive(Clone, Copy,Pod, Zeroable,Default)]
        },
        src: "
#version 450


#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

layout(local_size_x = 128, local_size_y = 1) in;


struct Vertex{
    float x;
	float y;
	float z;
    
    float r;
    float g;
    float b;
    float a;
};


layout(set = 0, binding = 0) uniform UniformData {
    mat4 world;
    mat4 view;
    mat4 proj;
    uvec2 img_size;
} uniforms;


layout(set = 0, binding = 1, std430) buffer point_data {
    Vertex vertices[];
};


layout (set = 0, binding = 2, std430) buffer framebuffer_data {
	uint64_t ssFramebuffer[];
};

void main() {
	uint globalID = gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) + gl_GlobalInvocationID.x;
    
    if (globalID >= vertices.length()){
        return;
    }
    
    Vertex v = vertices[globalID];

    mat4 uTransform = uniforms.proj * uniforms.view * uniforms.world;

	vec4 pos = uTransform * vec4(v.x, v.y, v.z, 1.0);
	pos.xyz = pos.xyz / pos.w;

	if(pos.w <= -1.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 || pos.y > 1.0){
		return;
	}

    vec2 imgPos = (pos.xy * 0.5 + 0.5) * uniforms.img_size;
	ivec2 pixelCoords = ivec2(imgPos);
	uint pixelID = pixelCoords.x + pixelCoords.y * uniforms.img_size.x;

    double depth = pos.w;
	int64_t u64Depth = int64_t(depth * 1000000.0lf);

    uint r_u = uint(v.r*255);
    uint g_u = uint(v.g*255);
    uint b_u = uint(v.b*255);
    uint a_u = uint(v.a*255);
    uint colors = r_u + (g_u << 8) + + (b_u << 16);

	int64_t val64 = (u64Depth << 24) | int64_t(colors);


	// 1 pixel
	uint64_t old = ssFramebuffer[pixelID];
	uint64_t oldDepth = (old >> 24);

	if(u64Depth < oldDepth){
		atomicMin(ssFramebuffer[pixelID], val64);
	}
}"
    }
}

mod cs_img {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450


#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable

layout(local_size_x = 16, local_size_y = 16) in;

layout (set = 0, binding = 0, std430) buffer framebuffer_data {
	uint64_t ssFramebuffer[];
};

layout(set=0, binding = 1, rgba8ui) uniform uimage2D uOutput;


uvec4 colorAt(int pixelID){
	uint64_t val64 = ssFramebuffer[pixelID];
	uint ucol = uint(val64 & 0x00FFFFFFUL);

	if(ucol == 0){
		return uvec4(0, 0, 0, 255);
	}

	vec4 color = 255.0 * unpackUnorm4x8(ucol);
	uvec4 icolor = uvec4(color);

	return icolor;
}

void main() {
    vec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = imageSize(uOutput);

	if(id.x >= imgSize.x){
		return;
	}

	ivec2 pixelCoords = ivec2(id);
	ivec2 sourceCoords = ivec2(id);
    sourceCoords.y = imgSize.y - sourceCoords.y;
	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

	
	uvec4 icolor = colorAt(pixelID);

	imageStore(uOutput, pixelCoords, icolor);

    uint64_t clear_color = 0xffffffffff000000UL + (255 << 0) + (0 << 8) + (0 << 16);
	ssFramebuffer[pixelID] = clear_color;

}"
    }
}

fn div_ceil(a: u32, b: u32) -> u32 {
    a / b + u32::from(a % b == 0)
}

fn main() {
    let opt = Opt::from_args();
    let filename = opt.input.as_os_str().to_str().unwrap();

    let octree: Octree<f32, f32> = {
        let in_file = File::open(filename).unwrap();

        let mut pb = ProgressBar::new(in_file.metadata().unwrap().len());

        let mut buf = BufReader::new(in_file);

        pb.message(&format!("decoding {}: ", filename));

        pb.set_units(pbr::Units::Bytes);
        let mut tee = TeeReader::new(&mut buf, &mut pb);

        let octree: Octree<f64, u8> = bincode::deserialize_from(&mut tee).unwrap();

        pb.finish_println("done!");

        octree.into()
    };

    let octree = Arc::new(octree);

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let mut event_loop = EventLoop::new(); // ignore this for now
    let surface = WindowBuilder::new()
        .with_title("puncTUM")
        .with_inner_size(PhysicalSize::new(800, 600))
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ext_buffer_device_address: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) =
        select_physical_device(&instance, &device_extensions, Some(surface.clone()));

    let (device, mut queues) = Device::new(
        // Which physical device to connect to.
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),

            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            enabled_features: Features {
                shader_int64: true,
                shader_float64: true,
                shader_buffer_int64_atomics: true,
                ..Features::none()
            },
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let swapchain_format = physical_device
        .surface_formats(&surface, Default::default())
        .unwrap()[0]
        .0;

    let render_pass = get_render_pass(device.clone(), swapchain_format);

    let mut viewport = Viewport::new(surface.window().inner_size().into());

    let mut frame = SurfaceFrame::new(
        surface.clone(),
        device.clone(),
        physical_device.clone(),
        render_pass.clone(),
        swapchain_format,
    );

    let vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex<f32, f32>]>> = unsafe {
        CpuAccessibleBuffer::uninitialized_array(
            device.clone(),
            octree.num_points(),
            BufferUsage::storage_buffer(),
            false,
        )
        .unwrap()
    };
    let mut offset = 0;
    let mut vertex_mapping: HashMap<u64, (usize, usize)> =
        HashMap::with_capacity(octree.num_octants() as usize);
    let mut vertices = vertex_buffer.write().unwrap();
    for octant in octree.into_iter() {
        let octant_size = octant.points().len();
        vertex_mapping.insert(octant.id(), (offset, octant_size));
        for (i, p) in octant.points().iter().enumerate() {
            vertices[offset + i] = *p;
        }
        offset += octant_size;
    }
    drop(vertices);

    let window_size = surface.window().inner_size();
    let aspect_ratio = window_size.width as f32 / window_size.height as f32;
    let mut camera = PerspectiveCamera::new(
        Point3::new(4.293682, 31.51273, -244.75063),
        Vector3::new(0.4617872, 0.0, 0.0),
        aspect_ratio,
    );
    // camera.adjust_znear_zfar(octree.bbox());

    let intermedite_image = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::storage_buffer(),
        false,
        (0..window_size.width * window_size.height).map(|_| 0u64),
    )
    .unwrap();

    let (compute_pipeline, set) = {
        let shader = cs::load(device.clone()).unwrap();

        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("failed to create compute pipeline");

        let uniform_data = cs::ty::UniformData {
            world: Matrix4::identity().into(),
            view: camera.view().clone().into(),
            proj: camera.projection().clone().into(),
            img_size: [window_size.width, window_size.height],
        };
        let uniform_buffer = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::uniform_buffer(),
            false,
            uniform_data,
        )
        .unwrap();

        let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, uniform_buffer.clone()),
                WriteDescriptorSet::buffer(1, vertex_buffer.clone()),
                WriteDescriptorSet::buffer(2, intermedite_image.clone()),
            ], // 0 is the binding
        )
        .unwrap();
        (compute_pipeline, set)
    };

    let target_img = StorageImage::new(
        device.clone(),
        vulkano::image::ImageDimensions::Dim2d {
            width: window_size.width,
            height: window_size.height,
            array_layers: 1,
        },
        Format::R8G8B8A8_UINT,
        [queue.family()],
    )
    .unwrap();

    let target_img_view = ImageView::new_default(target_img.clone()).unwrap();

    let (compute_pipeline_img, set_img) = {
        let shader = cs_img::load(device.clone()).unwrap();

        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("failed to create compute pipeline");

        let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
        let set_img = PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, intermedite_image.clone()),
                WriteDescriptorSet::image_view(1, target_img_view.clone()),
            ],
        )
        .unwrap();

        (compute_pipeline, set_img)
    };
    let mut camera_controller = CameraController::new(50., 1.);

    let mut last_update_inst = Instant::now();
    let mut last_mouse_position: Option<PhysicalPosition<f64>> = None;
    let mut mouse_pressed = false;
    event_loop.run_return(move |event, _, control_flow| match event {
        Event::WindowEvent { event, window_id } if window_id == surface.window().id() => {
            match event {
                WindowEvent::Resized(size) => {
                    viewport.resize(size.into());
                    camera.set_aspect_ratio(size.width as f32 / size.height as f32);
                    frame.force_recreate();
                }
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::CursorMoved {
                    position: new_pos, ..
                } => {
                    if mouse_pressed {
                        if let Some(last_pos) = last_mouse_position {
                            camera_controller.process_mouse(
                                (-(new_pos.x - last_pos.x) as f32).into(),
                                (-(new_pos.y - last_pos.y) as f32).into(),
                            );
                        }
                        last_mouse_position = Some(new_pos);
                    }
                }
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                } => match state {
                    ElementState::Pressed => {
                        mouse_pressed = true;
                    }
                    ElementState::Released => {
                        mouse_pressed = false;
                        last_mouse_position = None;
                    }
                },
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(key),
                            state,
                            ..
                        },
                    ..
                } => {
                    camera_controller.process_keyboard(key, state);
                }
                _ => (),
            }
        }
        Event::RedrawEventsCleared => {
            frame.recreate_if_necessary();

            let time_since_last_frame = last_update_inst.elapsed();
            last_update_inst = Instant::now();
            camera_controller.update_camera(&mut camera, time_since_last_frame);
            surface.window().request_redraw();
        }
        Event::RedrawRequested(..) => {
            frame.render_fn(queue.clone(), |acquire_future, img| {
                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let max_groups = device
                    .physical_device()
                    .properties()
                    .max_compute_work_group_count[0];

                let mut num_groups_x = div_ceil(octree.num_points() as u32, 128);
                let mut num_groups_y = 1;
                if num_groups_x > max_groups {
                    num_groups_y = div_ceil(num_groups_x, max_groups);
                    num_groups_x = max_groups;
                }

                builder
                    .bind_pipeline_compute(compute_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        compute_pipeline.layout().clone(),
                        0, // 0 is the index of our set
                        set.clone(),
                    )
                    .dispatch([num_groups_x, num_groups_y, 1])
                    .unwrap();

                let [width, height] = img.image().swapchain().image_extent();

                builder
                    .bind_pipeline_compute(compute_pipeline_img.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        compute_pipeline_img.layout().clone(),
                        0, // 0 is the index of our set
                        set_img.clone(),
                    )
                    .dispatch([(width + 15) / 16, (height + 15) / 16, 1])
                    .unwrap();

                builder
                    .copy_image(
                        target_img.clone(),
                        [0, 0, 0],
                        0,
                        0,
                        img.image().clone(),
                        [0, 0, 0],
                        0,
                        0,
                        [width, height, 1],
                        1,
                    )
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                acquire_future
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .boxed()
            });
        }
        _ => (),
    });
}
