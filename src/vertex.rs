#[derive(Default, Copy, Clone)]
pub struct Vertex {
    pub(crate) position: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position);
