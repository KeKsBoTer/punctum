#version 450

// 32 bits floating point PI
const float PI = 3.14159265358979323846264338327950288;

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;


layout(set = 0, binding = 0) uniform UniformData {
    mat4 world;
    mat4 view;
    mat4 proj;
    uint point_size;
    float zNear;
    float zFar;
} uniforms;

layout(set = 1, binding = 0) uniform sampler2DArray img_shs;
layout(set = 1, binding = 1) buffer SphericalCoefficients{
    vec4 coefs[];
} sh;


out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

layout(location = 0) out vec4 vertex_color;

vec4 sh_color(vec2 angle){

    vec2 img_size = vec2(imageSize(img_out));
    vec2 img_pos = vec2(angle.x / PI, angle.y / (2*PI));
    int num_sh = min(textureSize(img_shs,0).z,sh.coefs.length());

    vec4 color = vec4(0);
    for(int i=0;i<num_sh;i++){
        vec4 coefs = sh.coefs[i];
        color += coefs * texture(img_shs, vec3(img_pos/img_size,i)).r;
    }
    return color;
}

void main() {
    vec3 pos = position;
    mat4 worldview = uniforms.view * uniforms.world;
    gl_Position = uniforms.proj * worldview * vec4(pos, 1.0);
    
    gl_PointSize = uniforms.point_size;

    vertex_color = color;
}