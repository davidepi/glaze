#version 460
layout(location = 0) in vec2 v_pos;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in vec4 v_col;
layout(push_constant) uniform PushConstant { vec2 scale; vec2 translate; } Pc;
out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) out struct { vec4 col; vec2 uv; } Out;

void main()
{
    Out.col = v_col;
    Out.uv = v_uv;
    gl_Position = vec4(v_pos * Pc.scale + Pc.translate, 0, 1);
}