#version 460
// #extension GL_EXT_scalar_block_layout : enable
// #extension GL_EXT_buffer_reference2 : require
// #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// 
// struct Vertex 
// {
//  vec3 pos;
//  vec4 color;
// };
// 
// 
// layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer Vertices {Vertex v[]; }; // Positions of an object

layout(location = 0) in vec3 position;
// layout(location = 1) in vec3 normal;
// layout(location = 2) in vec4 color;
layout(location = 1) in vec4 color;

layout(set = 0, binding = 0) uniform UniformData {
    mat4 world;
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    uint point_size;
} uniforms;


// layout(set = 0, binding = 1, scalar) uniform ObjDesc { uint64_t vertexAddress; } objDesc;


out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};


layout(location = 0) out vec4 vertex_color;
layout(location = 1) out float pointSize;

void main() {

    // Vertices vertices = Vertices(objDesc.vertexAddress);

    vec4 pos = vec4(position,1);
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * pos;
    
    gl_PointSize = uniforms.point_size;

    vertex_color = color;//vertices.v[gl_VertexIndex].color;

    pointSize = gl_PointSize;
}