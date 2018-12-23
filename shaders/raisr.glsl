#version 430

//#pragma optionNV(unroll all)
//#pragma optionNV(fastmath off)
//#pragma optionNV(fastprecision off)

#define BLOCK_DIM
#define IMAGE_KERNEL_HALF_SIZE 5
#define GRADIENT_KERNEL_HALF_SIZE 4
#define ALIGNED_PATCH_ELEMENT_SIZE 4
#define ALIGNED_PATCH_VEC_SIZE 132
#define ALIGNED_PATCH_VEC_ELEMENTS (ALIGNED_PATCH_VEC_SIZE / ALIGNED_PATCH_ELEMENT_SIZE)

#define M_PI 3.1415926535897932384626433832795

#define Qangle 24
#define Qstrength 3
#define Qcoherence 3

#define EQ(a, b) (length(abs(a - b)) < 0.1)

precision highp float;

layout(local_size_x = BLOCK_DIM, local_size_y = BLOCK_DIM, local_size_z = 1) in;

uniform sampler2D lr_image;
uniform usamplerBuffer filterbank;
uniform sampler1D bounds;
uniform usampler2D hash_image;
uniform bool hash_image_enabled;
uniform uint R;
layout(RGBA8) uniform image2D hr_image;

shared float bilinear_data[BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE][
    BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE];
shared vec2 bilinear_chroma_data[BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE][
    BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE];
shared float gradient_xx[BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE][
    BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE];
shared float gradient_xy[BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE][
    BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE];
shared float gradient_yy[BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE][
    BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE];

#define GRADIENT_GATHER

#line 41

vec4 to_ycbcr(in vec4 inp) {
    return vec4(dot(inp, vec4(0.299, 0.587, 0.114, 0.0)),
                dot(inp, vec4(-0.168736, -0.331264, 0.500, 0.0)),
                dot(inp, vec4(0.5, -0.418688, -0.081312, 0.0)),
                1.0);
}

vec4 from_ycbcr(in vec4 inp) {
    return vec4(dot(inp, vec4(1.0, 0.0, 1.402, 0.0)),
                dot(inp, vec4(1.0, -0.344136, -0.714136, 0.0)),
                dot(inp, vec4(1.0, 1.772, 0.0, 0.0)),
                1.0);
}

vec4 bilinear_filter(in ivec2 hr_location) {
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

void load_bilinear_into_shared_tiled() {
    uvec2 block_origin = gl_WorkGroupID.xy * BLOCK_DIM;
    uvec2 block_dims = uvec2(BLOCK_DIM);
    uvec2 thread_idx = uvec2(gl_LocalInvocationID.xy);
    uvec2 margins = uvec2(IMAGE_KERNEL_HALF_SIZE);
    uvec2 upper_left = block_origin - margins;
    uvec2 lower_right = block_origin + BLOCK_DIM + margins;

    uint overall_dim_bilinear = BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE;
    uint overall_dim = BLOCK_DIM + 2 * IMAGE_KERNEL_HALF_SIZE;

    for (uint a_i = 0; a_i <= overall_dim_bilinear / BLOCK_DIM; a_i++) {
        for (uint a_j = 0; a_j <= overall_dim_bilinear / BLOCK_DIM; a_j++) {
            uvec2 mem_offset = uvec2(a_i, a_j) * BLOCK_DIM + thread_idx;
            ivec2 coords = ivec2(upper_left + mem_offset);
            if (coords.x < lower_right.x && coords.y < lower_right.y) {
                vec4 ycbcr = to_ycbcr(bilinear_filter(coords));
                bilinear_data[mem_offset.x][mem_offset.y] = ycbcr.x;
                bilinear_chroma_data[mem_offset.x][mem_offset.y] =
                    vec2(ycbcr.y, ycbcr.z);
            }
        }
    }

    barrier();

    lower_right = lower_right - 1;

    for (uint a_i = 0; a_i <= overall_dim / BLOCK_DIM; a_i++) {
        for (uint a_j = 0; a_j <= overall_dim / BLOCK_DIM; a_j++) {
            uvec2 mem_offset = uvec2(a_i, a_j) * BLOCK_DIM + thread_idx;
            uvec2 bmo = mem_offset + 1;
            uvec2 coords = bmo + upper_left;
            if (coords.x < lower_right.x && coords.y < lower_right.y) {
                vec4 col_0 = vec4(bilinear_data[bmo.x - 1][bmo.y - 1],
                                  bilinear_data[bmo.x - 1][bmo.y],
                                  bilinear_data[bmo.x - 1][bmo.y + 1],
                                  0);
                vec4 col_1 = vec4(bilinear_data[bmo.x][bmo.y - 1],
                                  bilinear_data[bmo.x][bmo.y],
                                  bilinear_data[bmo.x][bmo.y + 1],
                                  0);
                vec4 col_2 = vec4(bilinear_data[bmo.x + 1][bmo.y - 1],
                                  bilinear_data[bmo.x + 1][bmo.y],
                                  bilinear_data[bmo.x + 1][bmo.y + 1],
                                  0);

                float sobel_x;
                sobel_x = dot(vec4(-1.0, -2.0, -1.0, 0.0), col_0);
                sobel_x += dot(vec4(1.0, 2.0, 1.0, 0.0), col_2);

                float sobel_y;
                sobel_y = dot(vec4(-1.0, 0.0, 1.0, 0.0), col_0);
                sobel_y += dot(vec4(-2.0, 0.0, 2.0, 0.0), col_1);
                sobel_y += dot(vec4(-1.0, 0.0, 1.0, 0.0), col_2);

                gradient_xx[mem_offset.x][mem_offset.y] = sobel_y * sobel_y;
                gradient_xy[mem_offset.x][mem_offset.y] = sobel_y * sobel_x;
                gradient_yy[mem_offset.x][mem_offset.y] = sobel_x * sobel_x;
            }
        }
    }

    barrier();
}

vec4 weight_gradient(uvec2 upper_left) {
    // The reason why this is the case is quite confusing
    uvec2 center = upper_left + GRADIENT_KERNEL_HALF_SIZE;
    uvec2 lower_right = upper_left + GRADIENT_KERNEL_HALF_SIZE * 2;

    // G @ W @ G.T
    vec4 undecomposed = vec4(0.0, 0.0, 0.0, 0.0);
    GRADIENT_GATHER()
    undecomposed.z = undecomposed.y;

    return vec4(undecomposed.x, undecomposed.y,
                undecomposed.z, undecomposed.w);
}

void eigendecomposition(in vec4 m, out vec2 lambda, out vec4 evec) {
    if (m.y * m.z <= 1e-20) {
        lambda = vec2(m.x, m.w);
        evec = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        float tr_2 = (m.x + m.w) / 2.0;
        float det = m.x * m.w - m.y * m.z;
        float S = sqrt(tr_2 * tr_2 - det);
        lambda = vec2(tr_2 + S, tr_2 - S);

        float a_d_2 = (m.x - m.w) / 2.0;
        float SS = sqrt(max(a_d_2 * a_d_2 + m.y * m.z, 0.0));

        if (a_d_2 < 0.0) {
            evec = vec4(m.z, a_d_2 - SS, -a_d_2 + SS, m.y);
        } else {
            evec = vec4(a_d_2 + SS, m.z, m.y, -a_d_2 - SS);
        }

        // Fix normalization some day
        float n1 = sqrt(evec.x * evec.x + evec.y * evec.y);
        float n2 = sqrt(evec.z * evec.z + evec.w * evec.w);

        evec = vec4(evec.x / n1, evec.y, evec.z / n2, evec.w);
    }
}

uvec4 hashkey(in uvec2 thread_idx, in uvec2 block_idx) {
    vec4 weighted_grad = weight_gradient(thread_idx);

    vec2 eval;
    vec4 evec;
    eigendecomposition(weighted_grad, eval, evec);

    float theta = atan(evec.z, evec.x);
    theta = theta < 0.0 ? (theta + M_PI) : theta;

    float lambda = eval.x;
    float sqrtlambda1 = sqrt(eval.x);
    float sqrtlambda2 = sqrt(eval.y);

    float u;
    if (sqrtlambda1 + sqrtlambda2 == 0.0) {
        u = 0.0;
    } else if (eval.x < 0.0 || eval.y < 0.0) {
        u = 0.33;
    } else {
        u = (sqrtlambda1 - sqrtlambda2) / (sqrtlambda1 + sqrtlambda2);
    }

    uint angle = uint(floor((theta / M_PI) * Qangle));
    uint strength = (lambda < 0.0001) ? 0 : (lambda > 0.001 ? 2 : 1);
    uint coherence = (u < 0.25) ? 0 : (u > 0.5 ? 2 : 1);

    angle = (angle > 23) ? 23 : (angle < 0 ? 0 : angle);

    uvec2 mod_val = (block_idx * BLOCK_DIM + thread_idx - IMAGE_KERNEL_HALF_SIZE);
    uint pixel_type = uint(mod(mod_val.x, R) * R) + uint(mod(mod_val.y, R));

    return uvec4(angle, strength, coherence, pixel_type);
}

float apply_filter(uvec4 key, uvec2 upper_left) {
    // base = pixel_type +
    //            coherence * R_2 +
    //            strength * Qcoherence * R_2 +
    //            angle * Qstrength * Qcoherence * R_2

    // bounds_offset = base
    // fb_offset = base * ALIGNED_PATCH_VEC_ELEMENTS

    /*uint base_offset = uint(dot(key, uvec4(Qstrength * Qcoherence * R * R,
                                           Qcoherence * R * R,
                                           R * R,
                                           1)));*/
    uint base_offset = key.x * Qstrength * Qcoherence * R * R +
                       key.y * Qcoherence * R * R + key.z * R * R + key.w;

    uint bounds_offset = base_offset;
    uint fb_offset = base_offset * ALIGNED_PATCH_VEC_ELEMENTS;

    vec2 minmax = texelFetch(bounds, int(bounds_offset), 0).rg;
    float min_val = minmax.x;
    float max_val = minmax.y;
    float span = max_val - min_val;
    uint fb_ptr = fb_offset;

    uvec2 center = upper_left + IMAGE_KERNEL_HALF_SIZE;
    uint kernel_offset = IMAGE_KERNEL_HALF_SIZE * 2 + 1;

    // Accounts for some alignment issues
    // NOTE: EXCLUSIVE BOUND
    uvec2 lower_right =
        uvec2(upper_left.x + kernel_offset - uint(mod(kernel_offset,
                ALIGNED_PATCH_ELEMENT_SIZE)),
              upper_left.y + kernel_offset);

    float accum = 0.0;

    for (uint j = upper_left.y; j < lower_right.y; j++) {
        for (uint i = upper_left.x; i < lower_right.x; i += 4) {
            vec4 filters = ((vec4(texelFetch(filterbank, int(fb_ptr)))
                             + 0.5) / 255.0) * span + min_val;
            vec4 seq = vec4(bilinear_data[i][j],
                            bilinear_data[i + 1][j],
                            bilinear_data[i + 2][j],
                            bilinear_data[i + 3][j]);

            accum += dot(seq, filters);
            fb_ptr++;
        }

        // TODO: Properly handle different cases here
        vec3 filters = ((vec3(texelFetch(filterbank, int(fb_ptr)))
                         + 0.5) / 255.0) * span + min_val;
        vec3 seq = vec3(bilinear_data[lower_right.x][j],
                        bilinear_data[lower_right.x + 1][j],
                        bilinear_data[lower_right.x + 2][j]);
        accum += dot(seq, filters);
        fb_ptr++;
    }

    return accum;
}

bool test_eigendecomposition() {
    vec2 eval;
    vec4 evec;

    eigendecomposition(vec4(5.0, 3.0, 3.0, 2.0), eval, evec);

    bool t1 = EQ(eval,
                 vec2(6.854101966249685, 0.1458980337503153))
              && EQ(evec,
                    vec4(0.8506508083520399, 3.0, 0.5257311121191336, -4.854101966249685));

    eigendecomposition(vec4(2.0, 3.0, 3.0, 4.0), eval, evec);

    bool t2 = EQ(eval,
                 vec2(6.16227766016838, -0.16227766016837952))
              && EQ(evec,
                    vec4(0.5847102846637648, -4.16227766016838, 0.8112421851755608, 3));

    return t1 && t2;
}

bool test_hashkey() {
    // TODO: Flesh out this test
    uvec4 result = hashkey(uvec2(0, 0), uvec2(0, 0));

    return true;
    //return (result.xyz == uvec3(7, 2, 1));
}

void run_tests() {
    bool passing = true;
    passing = passing && test_eigendecomposition();
    passing = passing && test_hashkey();
    uvec3 index = gl_GlobalInvocationID;

    if (passing) {
        imageStore(hr_image, ivec2(index.xy), vec4(0.0, 1.0, 0.0, 1.0));
    } else {
        imageStore(hr_image, ivec2(index.xy), vec4(0.0, 0.0, 0.0, 1.0));
    }
}

void main() {
    uvec3 index = gl_GlobalInvocationID;
    //vec4 data = bilinear_filter(ivec2(index.xy));
    load_bilinear_into_shared_tiled();

    ivec2 offset = ivec2(gl_LocalInvocationID.xy) + IMAGE_KERNEL_HALF_SIZE;
    //vec4 sobel = vec4(vec2(gradient_xx[offset.x][offset.y], gradient_xy[offset.x][offset.y]), 0.0, 1.0);

    /*vec4 weighted_grad = weight_gradient(gl_LocalInvocationID.xy);
    vec2 eval;
    vec4 evec;
    eigendecomposition(weighted_grad, eval, evec);*/

    uvec4 key;
    if (hash_image_enabled) {
        uvec2 mod_val = (gl_WorkGroupID.xy * BLOCK_DIM + gl_LocalInvocationID.xy -
                         IMAGE_KERNEL_HALF_SIZE);
        uint pixel_type = uint(mod(mod_val.x, R) * R) + uint(mod(mod_val.y, R));
        key = texelFetch(hash_image, ivec2(index.xy), 0);
        key.w = pixel_type;
    } else {
        key = hashkey(gl_LocalInvocationID.xy, gl_WorkGroupID.xy);
    }

    float accum = apply_filter(key, gl_LocalInvocationID.xy);

    vec4 vis = vec4(key) / vec4(Qangle, Qstrength, Qcoherence, R * R);
    vis.w = 1.0;

    vec2 chroma = bilinear_chroma_data[offset.x][offset.y];

    //imageStore(hr_image, ivec2(index.xy), sobel);
    //imageStore(hr_image, ivec2(index.xy), vec4(vis.x, vis.y, vis.z, 1.0));
    //imageStore(hr_image, ivec2(index.xy), vec4(accum, accum, accum, 1.0));
    imageStore(hr_image, ivec2(index.xy), from_ycbcr(vec4(accum, chroma.r, chroma.g,
               1.0)));

    /*if (mod(gl_LocalInvocationID.x, BLOCK_DIM) == 0 ||
            mod(gl_LocalInvocationID.y, BLOCK_DIM) == 0) {
        imageStore(hr_image, ivec2(index.xy), vec4(0.0, 1.0, 0.0, 1.0));
        }*/

    // Verify that filterbank is loaded
    /*int overall = imageSize(hr_image).x * int(index.y) + int(index.x);

    imageStore(hr_image, ivec2(index.xy),
    vec4(texelFetch(filterbank, overall).xyz, 255.0) / 255.0);*/

    //run_tests();
}
