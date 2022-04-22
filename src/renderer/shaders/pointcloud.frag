#version 450

layout(location = 0) in vec4 vertex_color;
layout(location = 0) out vec4 f_color;

// void main() {
//     // turn points into circles
//     float d = length(vec2(0.5)-gl_PointCoord);
//     if(d>0.5){
//         discard;
//     }
//     if(d > 0.4){
//         f_color = vec4(mix(vertex_color,vec3(0.),0.8),1.0);
//     }else{
//         f_color = vec4(vertex_color,1.0);
//     }
// }

void main() {
    // turn points into circles
    float d = length(vec2(0.5)-gl_PointCoord);
    if(d>0.5){
        discard;
    }
    f_color = vertex_color;
}