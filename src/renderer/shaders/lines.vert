#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in uint color;

layout(set = 0, binding = 0) uniform UniformData {
    mat4 world;
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    uint point_size;
    bool highlight_sh;
} uniforms;

layout(location = 0) out vec3 vertex_color;

void main() {

    vec3 color_n = unpackUnorm4x8(color).rgb;
    vec4 pos = vec4(position,1);
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * pos;
    // TODO do this in projection matrix
    gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
    

    vertex_color = color_n;
}