#version 450

// 32 bits floating point PI
const float PI = 3.14159265358979323846264338327950288;

layout(constant_id = 0) const int num_coefs = (10+1)*(10+1);

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 sh_coefs[num_coefs];


layout(set = 0, binding = 0) uniform UniformData {
    mat4 world;
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    uint point_size;
    float zNear;
    float zFar;
} uniforms;

layout(set = 1, binding = 0) uniform sampler2DArray img_shs;


out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

layout(location = 0) out vec4 vertex_color;

vec4 sh_color(vec2 angle){

    ivec3 img_size = textureSize(img_shs,0);
    vec2 img_pos = vec2(angle.x / PI, angle.y / (2*PI));
    int num_sh = min(img_size.z,num_coefs);

    vec4 color = vec4(0);
    for(int i=0;i<num_sh;i++){
        vec4 coefs = sh_coefs[i];
        color += coefs * texture(img_shs, vec3(img_pos/vec2(img_size.xy),i)).r;
    }
    return color;
}

void main() {
    vec4 pos = vec4(position, 1.0);
    // absolute position within world
    vec4 world_pos = uniforms.world * pos;
    
    // position relative to camera
    vec4 camera_pos = uniforms.view * world_pos;
    gl_Position = uniforms.proj * camera_pos;
    
    gl_PointSize = uniforms.point_size;

    vec3 diff = normalize(world_pos.xyz - uniforms.camera_pos);
    vec2 angle = vec2(acos(diff.z),atan(diff.y,diff.x) + PI);

    vertex_color = sh_color(angle);
}