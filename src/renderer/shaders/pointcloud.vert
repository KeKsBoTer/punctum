#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(set = 0, binding = 0) uniform UniformData {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;


out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

layout(location = 0) out vec3 vertex_color;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);

    // angle to camera
    float angle = dot((worldview * vec4(0,0,1.0,1.0)).xyz,normal);

    gl_PointSize = mix(5.0,10., (angle+1.0)/2.0);
    // gl_PointSize = (1/gl_Position.z);

    vertex_color = color; //*  (angle+1.0)/2.0;
}