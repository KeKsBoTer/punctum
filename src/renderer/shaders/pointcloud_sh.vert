#version 460

// 32 bits floating point PI
const float PI = 3.14159265358979323846264338327950288;

//layout(constant_id=0)
const uint num_coefs = 25;


layout(location = 0) in vec3 position;
layout(location = 1) in float size;
layout(location = 2) in vec4 coefficients[num_coefs];

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

vec4 sh_color(vec2 angle){

    ivec3 img_size = textureSize(img_shs,0);
    vec2 img_pos = vec2(angle.y / (2*PI),angle.x / PI);
    uint num_sh = min(uint(img_size.z),num_coefs);

    vec4 color = vec4(0);
    for(int i=0;i<num_sh;i++){
        vec4 coefs = coefficients[i];
        color += coefs * texture(img_shs, vec3(img_pos,i)).r;
    }
    return color;
}

void main() {
    // absolute position within world
    vec4 world_pos = uniforms.world * vec4(position,1);
    
    // position relative to camera
    vec4 camera_pos = uniforms.view * world_pos;
    gl_Position = uniforms.proj * camera_pos;

    // depth correction
    // use use a OpenGL style projection matrix
    // this matrix normalizes the depth to [-1,1]
    // but vulkan uses [0,1] for the depth buffer, so we correct for this
    // TODO fix this in projection matrix calculation
    gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;

    vec3 d = world_pos.xyz - uniforms.camera_pos;
    float cam_distance = length(d);
    
    float screen_p = 2* atan(size/(2*cam_distance));
    float fov = PI/2;

    gl_PointSize = screen_p/fov * 1080;
    pointSize = gl_PointSize;
    vertex_pos = gl_Position;

    if(uniforms.highlight_sh){
        vertex_color = vec4(1.,0.,0.,1.);
        return;
    }

    vec3 camera_normal = d/cam_distance;
    vec2 angle = vec2(acos(camera_normal.z),atan(-camera_normal.y,camera_normal.x) + PI);

    vec4 dir_color = sh_color(angle);

    if (uniforms.transparency)
        // alpha correction
        // we calculated the alpha value on the whole image 
        // but only the circle is relevant so we need to correct for it
        // TODO: change this in sh coef calculation (model & dataset)
        dir_color.a /= 0.5*0.5*PI;
    else
        dir_color.a = 1;
    
    vertex_color = dir_color;
}