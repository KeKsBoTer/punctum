#version 450

layout(location = 0) in vec3 position;

out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

void main() {
    gl_Position = vec4(position.xy, 0.0, 1.0);
    gl_PointSize = 20.0;
}