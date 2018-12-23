#!/usr/bin/env python3

import numpy as np
import sys

half_gradient = 4

iteration = '''
undecomposed.x += gradient_xx[upper_left.x + {i}][upper_left.y + {j}] * {gaussians}f; \\
undecomposed.y += gradient_xy[upper_left.x + {i}][upper_left.y + {j}] * {gaussians}f; \\
undecomposed.z += gradient_yy[upper_left.x + {i}][upper_left.y + {j}] * {gaussians}f; \\'''

iteration_improved = '''
undecomposed += vec3(                                                      \\
    gradient_xx[upper_left.x + {i}][upper_left.y + {j}],                   \\
    gradient_xy[upper_left.x + {i}][upper_left.y + {j}],                   \\
    gradient_yy[upper_left.x + {i}][upper_left.y + {j}]) * {gaussians}f;   \\'''


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


def unroll_loops():
    dim = half_gradient * 2 + 1
    weighting = gaussian2d([dim, dim], 2)

    source = '''#define GRADIENT_GATHER() \\\nvec3 gaussians; \\'''

    for i in range(dim):
        for j in range(dim):
            source += iteration_improved.format(i=i,
                                                j=j, gaussians=weighting[i][j])

    source = source[: -2] + '\n'

    with open('shaders/gradient_gather.glsl', 'w') as h:
        h.write(source)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        half_gradient = int(sys.argv[1])
    unroll_loops()
