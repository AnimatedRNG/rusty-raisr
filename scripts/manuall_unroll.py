#!/usr/bin/env python3

import numpy as np
import sys

half_gradient = 4
half_filter = 5

gradient_iteration = '''
undecomposed += gradient[upper_left.x + {i}][upper_left.y + {j}] * {gaussians}f;   \\'''

filtering_iteration_aligned = '''
filters = ((f16vec4(texelFetch(filterbank, fb_offset + {index}))            \\
              + 0.5) / 255.0) * span + min_val;                             \\
seq = f16vec4(bilinear_data[upper_left.x + {i}][upper_left.y + {j}],        \\
          bilinear_data[upper_left.x + {i} + 1][upper_left.y + {j}],        \\
          bilinear_data[upper_left.x + {i} + 2][upper_left.y + {j}],        \\
          bilinear_data[upper_left.x + {i} + 3][upper_left.y + {j}]);       \\
accum += dot(seq, filters);                                                 \\
                                                                            \\'''

filtering_iteration_unaligned = '''
filters_odd = ((f16vec3(texelFetch(filterbank, fb_offset + {index}))             \\
              + 0.5) / 255.0) * span + min_val;                                  \\
seq_odd = f16vec3(bilinear_data[upper_left.x + {i}][upper_left.y + {j}],         \\
                  bilinear_data[upper_left.x + {i} + 1][upper_left.y + {j}],     \\
                  bilinear_data[upper_left.x + {i} + 2][upper_left.y + {j}]);    \\
accum += dot(seq_odd, filters_odd);                                              \\
                                                                                 \\'''


def gaussian2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])

    From github.com/movehand/raisr
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def unroll_gradient():
    dim = half_gradient * 2 + 1
    weighting = gaussian2d([dim, dim], 2)

    gradient_source = '''#define GRADIENT_GATHER() \\\nvec3 gaussians; \\'''

    for i in range(dim):
        for j in range(dim):
            gradient_source += gradient_iteration.format(i=i,
                                                         j=j, gaussians=weighting[i][j])

    gradient_source = gradient_source[: -2] + '\n'

    with open('shaders/gradient_gather.glsl', 'w') as h:
        h.write(gradient_source)


def unroll_filter():
    dim = half_filter * 2 + 1

    filter_source = '''
#define ACCUMULATE_FILTER() \\
float16_t accum = 0.0;      \\
f16vec4 filters;            \\
f16vec4 seq;                \\
f16vec3 filters_odd;        \\
f16vec3 seq_odd;            \\'''

    index = 0
    end_index = dim - (dim % 4)

    for j in range(dim):
        for i in range(0, end_index, 4):
            filter_source += filtering_iteration_aligned.format(
                i=i, j=j, index=index)
            index += 1
        filter_source += filtering_iteration_unaligned.format(
            i=end_index, j=j, index=index)
        index += 1

    with open('shaders/filter.glsl', 'w') as h:
        h.write(filter_source)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        half_gradient = int(sys.argv[1])
    if len(sys.argv) > 2:
        half_filter = int(sys.argv[2])
    unroll_gradient()
    unroll_filter()
