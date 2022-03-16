#version 450

layout(location = 0) in vec3 position;

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

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
    gl_PointSize = 20.0;
}