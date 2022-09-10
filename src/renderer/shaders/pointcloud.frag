#version 460

layout(location = 0) in vec4 vertex_color;
layout(location = 1) in vec4 vertex_pos;
layout(location = 2) in float pointSize;

layout(location = 0) out vec4 f_color;

void main() {
    if(pointSize>3){
        // turn points into circles
        vec2 diff = vec2(0.5)-gl_PointCoord;
        float d = dot(diff,diff);
        if(d>0.25){
            discard;
        }
    }
    f_color = vertex_color;
}