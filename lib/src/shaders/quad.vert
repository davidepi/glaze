#version 460

layout (location = 0) out vec2 out_uv;

#ifdef SKYDOME
// used in skydome calculation
layout(push_constant) uniform readonly pc {
  vec2 uvs[3];
} UVs;
#endif

void main()
{
  // generates a triangle that covers up the entire viewport.
  // the generated coordinates are
  // 0 -> UV(0,0), Pos(-1,-1) (top-left)
  // 1 -> UV(2,0), Pos( 3,-1) (top-right)
  // 2 -> UV(0,2), Pos(-1, 3) (bottom-left)
  //
  // Even though the bottom right of the viewport is (1, 1) some portions of this triangle will be
  // clipped, hence the UV>1 and Pos>1
  //
  // explained in detail here:
  // https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers
	out_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(out_uv * 2.0f - 1.0f, 0.1f, 1.0f);

#ifdef SKYDOME
  // Use different UVs for skydome rendering
  out_uv = UVs.uvs[gl_VertexIndex];
#endif
}
