
#define ACCUMULATE_FILTER() \
float16_t accum = 0.0;      \
f16vec4 filters;            \
f16vec4 seq;                \
f16vec3 filters_odd;        \
f16vec3 seq_odd;            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 0))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 0][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 0][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 0][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 0][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 1))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 0][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 0][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 0][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 0][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 2))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 0][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 0][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 0][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 3))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 1][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 1][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 1][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 1][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 4))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 1][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 1][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 1][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 1][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 5))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 1][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 1][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 1][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 6))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 2][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 2][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 2][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 2][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 7))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 2][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 2][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 2][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 2][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 8))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 2][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 2][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 2][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 9))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 3][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 3][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 3][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 3][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 10))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 3][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 3][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 3][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 3][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 11))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 3][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 3][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 3][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 12))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 4][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 4][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 4][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 4][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 13))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 4][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 4][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 4][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 4][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 14))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 4][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 4][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 4][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 15))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 5][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 5][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 5][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 5][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 16))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 5][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 5][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 5][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 5][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 17))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 5][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 5][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 5][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 18))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 6][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 6][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 6][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 6][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 19))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 6][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 6][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 6][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 6][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 20))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 6][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 6][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 6][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 21))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 7][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 7][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 7][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 7][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 22))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 7][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 7][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 7][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 7][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 23))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 7][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 7][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 7][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 24))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 8][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 8][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 8][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 8][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 25))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 8][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 8][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 8][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 8][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 26))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 8][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 8][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 8][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 27))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 9][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 9][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 9][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 9][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 28))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 9][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 9][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 9][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 9][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 29))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 9][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 9][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 9][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 30))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 10][upper_left.x + 0],        \
          bilinear_data[upper_left.y + 10][upper_left.x + 0 + 1],        \
          bilinear_data[upper_left.y + 10][upper_left.x + 0 + 2],        \
          bilinear_data[upper_left.y + 10][upper_left.x + 0 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters = ((f16vec4(texelFetch(filterbank, fb_offset + 31))            \
              + 0.5) / 255.0) * span + min_val;                             \
seq = f16vec4(bilinear_data[upper_left.y + 10][upper_left.x + 4],        \
          bilinear_data[upper_left.y + 10][upper_left.x + 4 + 1],        \
          bilinear_data[upper_left.y + 10][upper_left.x + 4 + 2],        \
          bilinear_data[upper_left.y + 10][upper_left.x + 4 + 3]);       \
accum += dot(seq, filters);                                                 \
                                                                            \
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + 32))             \
              + 0.5) / 255.0) * span + min_val;                                  \
seq_odd = f16vec3(bilinear_data[upper_left.y + 10][upper_left.x + 8],         \
                  bilinear_data[upper_left.y + 10][upper_left.x + 8 + 1],     \
                  bilinear_data[upper_left.y + 10][upper_left.x + 8 + 2]);    \
accum += dot(seq_odd, filters_odd);                                              \
                                                                                 \