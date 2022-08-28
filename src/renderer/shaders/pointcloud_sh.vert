#version 460

// 32 bits floating point PI
const float PI = 3.14159265358979323846264338327950288;

//layout(constant_id=0)
const uint num_coefs = 25;


layout(location = 0) in vec3 position;
layout(location = 1) in float size;
layout(location = 2) in vec3 coefficients[num_coefs];

layout(set = 0, binding = 0) uniform UniformData {
    mat4 world;
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    uint point_size;
    bool highlight_sh;
    bool transparency;
} uniforms;

layout(set = 1, binding = 0) uniform sampler2DArray img_shs;


out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

layout(location = 0) out vec4 vertex_color;
layout(location = 1) out vec4 vertex_pos;
layout(location = 2) out float pointSize;

vec3 sh_color(vec2 angle){

    ivec3 img_size = textureSize(img_shs,0);
    vec2 img_pos = vec2(angle.y / (2*PI),angle.x / PI);
    uint num_sh = min(uint(img_size.z),num_coefs);

    vec3 color = vec3(0);
    for(int i=0;i<num_sh;i++){
        vec3 coefs = coefficients[i];
        color += coefs * texture(img_shs, vec3(img_pos,i)).r;
    }
    return color;
}

float calc_point_size(vec2 screen_pos,float cam_distance){
    float a = length(vec2(screen_pos.x*16/9,screen_pos.y)-0.5);
    
    float screen_p = 2* atan(size/(2*cam_distance));
    float fovy = PI;

    return 2 * screen_p/fovy * 1080; //* (1 +  2*tan(0.5*a*fovy/2));
}

void main() {
    // absolute position within world
    vec4 world_pos = uniforms.world * vec4(position,1);
    
    // position relative to camera
    vec4 camera_pos = uniforms.view * world_pos;
    gl_Position = uniforms.proj * camera_pos;

    // depth correction
    // we use a OpenGL style projection matrix
    // this matrix normalizes the depth to [-1,1]
    // but vulkan uses [0,1] for the depth buffer, so we correct for this
    // TODO fix this in projection matrix calculation
    gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;

    vec2 screen_pos = gl_Position.xy/gl_Position.w;
    vec3 d = camera_pos.xyz;
    float cam_distance = length(d);

    gl_PointSize = calc_point_size(screen_pos,cam_distance);
    pointSize = gl_PointSize;
    vertex_pos = gl_Position;

    if(uniforms.highlight_sh){
        vertex_color = vec4(1.,0.,0.,1.);
        return;
    }

    vec3 camera_normal = d/cam_distance;
    vec2 angle = vec2(acos(camera_normal.z),atan(-camera_normal.y,camera_normal.x) + PI);

    vec3 dir_color = sh_color(angle);
    
    vertex_color = vec4(dir_color,1.);
}