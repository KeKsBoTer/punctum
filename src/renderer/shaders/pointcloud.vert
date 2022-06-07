#version 450

layout(location = 0) in vec3 position;
// layout(location = 1) in vec3 normal;
// layout(location = 2) in vec4 color;
layout(location = 1) in vec4 color;

layout(set = 0, binding = 0) uniform UniformData {
    mat4 world;
    mat4 view;
    mat4 proj;
    float point_size;
    float zNear;
    float zFar;
} uniforms;


out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

layout(location = 0) out vec4 vertex_color;
layout(location = 1) out float zNear;
layout(location = 2) out float zFar;

void main() {
    vec3 pos = position;
    pos.yz = pos.zy;
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * vec4(pos, 1.0);

    float d = clamp(0,1, gl_Position.z / uniforms.zFar);

    gl_PointSize =  mix(uniforms.point_size,3,d);

    vertex_color = color;

    zNear = uniforms.zNear;
    zFar = uniforms.zFar;
}