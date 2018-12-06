#version 430

#define LOCAL_SIZE_X
#define LOCAL_SIZE_Y

precision highp float;

layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = 1) in;

uniform sampler2D lr_image;
uniform uint R;
layout(RGBA8) uniform image2D hr_image;

vec4 to_ycbcr(vec4 inp) {
    return vec4(dot(inp, vec4(0.299, 0.587, 0.114, 0.0)),
                dot(inp, vec4(-0.168736, -0.331264, 0.500, 0.0)),
                dot(inp, vec4(0.5, -0.418688, -0.081312, 0.0)),
                0.0);
}

vec4 from_ycbcr(vec4 inp) {
    return vec4(dot(inp, vec4(1.0, 0.0, 1.402, 0.0)),
                dot(inp, vec4(1.0, -0.344136, -0.714136, 0.0)),
                dot(inp, vec4(1.0, 1.772, 0.0, 0.0)),
                0.0);
}

vec4 bilinear_filter(ivec2 hr_location) {
    ivec2 lr_location = hr_location / int(R);
    vec4 ul = texelFetch(lr_image, lr_location, 0);
    vec4 ur = texelFetch(lr_image, lr_location + ivec2(1, 0), 0);
    vec4 dl = texelFetch(lr_image, lr_location + ivec2(0, 1), 0);
    vec4 dr = texelFetch(lr_image, lr_location + ivec2(1, 1), 0);

    vec2 f = fract(vec2(hr_location) / float(R));
    vec4 tA = mix(ul, ur, f.x);
    vec4 tB = mix(dl, dr, f.x);

    return mix(tA, tB, f.y);
}

void main() {
    uvec3 index = gl_GlobalInvocationID;
    vec4 data = bilinear_filter(ivec2(index.xy));
    imageStore(hr_image, ivec2(index.xy), data);
}
