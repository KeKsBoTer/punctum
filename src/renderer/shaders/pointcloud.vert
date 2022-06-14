#version 450

layout(location = 0) in vec3 position;
// layout(location = 1) in vec3 normal;
// layout(location = 2) in vec4 color;
layout(location = 1) in vec4 color;

layout(set = 0, binding = 0) uniform UniformData {
    mat4 world;
    mat4 view;
    mat4 proj;
    uint point_size;
    float zNear;
    float zFar;
} uniforms;


out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};


layout(location = 0) out vec4 vertex_color;
// layout(location = 1) out float zNear;
// layout(location = 2) out float zFar;
// layout(location = 3) out float pointSize;

void main() {
    vec3 pos = position;
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * vec4(pos, 1.0);
    
    gl_PointSize = uniforms.point_size;

    vertex_color = color;

    // pointSize = gl_PointSize;
    // zNear = uniforms.zNear;
    // zFar = uniforms.zFar;
}