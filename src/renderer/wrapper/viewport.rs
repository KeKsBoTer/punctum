use vulkano::pipeline::graphics::viewport::Viewport as VulkanViewport;

#[derive(Debug, Clone)]
pub struct Viewport {
    vp: VulkanViewport,
}

impl Viewport {
    pub fn new(size: [u32; 2]) -> Self {
        Viewport {
            vp: Viewport::build_vulkan_viewport(size),
        }
    }

    // build viewport with y flip
    // see: https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/
    fn build_vulkan_viewport(size: [u32; 2]) -> VulkanViewport {
        VulkanViewport {
            origin: [0.0, size[1] as f32],
            dimensions: [size[0] as f32, -(size[1] as f32)],
            depth_range: 0. ..1.0,
        }
    }

    pub fn resize(&mut self, size: [u32; 2]) {
        self.vp = Viewport::build_vulkan_viewport(size);
    }

    pub fn size(&self) -> [f32; 2] {
        [self.vp.dimensions[0], self.vp.dimensions[1].abs()]
    }
}

impl From<Viewport> for VulkanViewport {
    fn from(viewport: Viewport) -> Self {
        viewport.vp
    }
}
