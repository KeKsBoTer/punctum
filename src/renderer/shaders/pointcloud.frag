#version 460

layout(location = 0) in vec4 vertex_color;
layout(location = 1) in vec4 vertex_pos;
layout(location = 2) in float pointSize;

layout(location = 0) out vec4 f_color;

void main() {
    if(pointSize>1){
        // turn points into circles
        float d = length(vec2(0.5)-gl_PointCoord);
        if(d>0.5){
            discard;
        }
    }
    f_color = vertex_color;
}