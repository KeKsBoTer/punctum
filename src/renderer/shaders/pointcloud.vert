#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(set = 0, binding = 0) uniform UniformData {
    mat4 world;
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    uint point_size;
    bool highlight_sh;
} uniforms;


out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};


layout(location = 0) out vec4 vertex_color;
layout(location = 1) out vec4 vertex_pos;
layout(location = 2) out float pointSize;

void main() {

    vec4 pos = vec4(position,1);
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * pos;
    // TODO do this in projection matrix
    gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
    
    gl_PointSize = uniforms.point_size;

    vertex_color = color;

    pointSize = gl_PointSize;
    vertex_pos = gl_Position;
}