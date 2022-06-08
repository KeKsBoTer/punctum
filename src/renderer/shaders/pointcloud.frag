#version 450

layout(location = 0) in vec4 vertex_color;
layout(location = 1) in float zNear;
layout(location = 2) in float zFar;
layout(location = 3) in float pointSize;

layout(location = 0) out vec4 f_color;

float linearize_depth(float d)
{
    return zNear * zFar / (zFar + d * (zNear - zFar));
}

float delinearize_depth(float d)
{
    return zFar*(zNear-d)/(d*(zNear-zFar));
}

void main() {
    // if(pointSize>1){
    //     // turn points into circles
    //     float d = length(vec2(0.5)-gl_PointCoord);
    //     if(d>0.5){
    //         discard;
    //     }
    // }
    f_color = vertex_color;
}