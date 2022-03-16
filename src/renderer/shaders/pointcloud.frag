#version 450

layout(location = 0) out vec4 f_color;

void main() {
    // turn points into circles
    float d = length(vec2(0.5)-gl_PointCoord);
    if(d>0.5){
        discard;
    }
    f_color = vec4(gl_PointCoord, 0.0, 1.0);
}