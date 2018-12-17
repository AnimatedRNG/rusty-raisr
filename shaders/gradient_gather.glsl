#define GRADIENT_GATHER() \
vec3 gaussians; \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 0][upper_left.y + 0],                   \
    gradient_xy[upper_left.x + 0][upper_left.y + 0],                   \
    gradient_yy[upper_left.x + 0][upper_left.y + 0]) * 0.0007634473286087523f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 0][upper_left.y + 1],                   \
    gradient_xy[upper_left.x + 0][upper_left.y + 1],                   \
    gradient_yy[upper_left.x + 0][upper_left.y + 1]) * 0.0018314149348447166f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 0][upper_left.y + 2],                   \
    gradient_xy[upper_left.x + 0][upper_left.y + 2],                   \
    gradient_yy[upper_left.x + 0][upper_left.y + 2]) * 0.0034215335484046386f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 0][upper_left.y + 3],                   \
    gradient_xy[upper_left.x + 0][upper_left.y + 3],                   \
    gradient_yy[upper_left.x + 0][upper_left.y + 3]) * 0.004978301937756899f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 0][upper_left.y + 4],                   \
    gradient_xy[upper_left.x + 0][upper_left.y + 4],                   \
    gradient_yy[upper_left.x + 0][upper_left.y + 4]) * 0.005641155139668815f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 0][upper_left.y + 5],                   \
    gradient_xy[upper_left.x + 0][upper_left.y + 5],                   \
    gradient_yy[upper_left.x + 0][upper_left.y + 5]) * 0.004978301937756899f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 0][upper_left.y + 6],                   \
    gradient_xy[upper_left.x + 0][upper_left.y + 6],                   \
    gradient_yy[upper_left.x + 0][upper_left.y + 6]) * 0.0034215335484046386f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 0][upper_left.y + 7],                   \
    gradient_xy[upper_left.x + 0][upper_left.y + 7],                   \
    gradient_yy[upper_left.x + 0][upper_left.y + 7]) * 0.0018314149348447166f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 0][upper_left.y + 8],                   \
    gradient_xy[upper_left.x + 0][upper_left.y + 8],                   \
    gradient_yy[upper_left.x + 0][upper_left.y + 8]) * 0.0007634473286087523f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 1][upper_left.y + 0],                   \
    gradient_xy[upper_left.x + 1][upper_left.y + 0],                   \
    gradient_yy[upper_left.x + 1][upper_left.y + 0]) * 0.0018314149348447166f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 1][upper_left.y + 1],                   \
    gradient_xy[upper_left.x + 1][upper_left.y + 1],                   \
    gradient_yy[upper_left.x + 1][upper_left.y + 1]) * 0.004393336040201352f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 1][upper_left.y + 2],                   \
    gradient_xy[upper_left.x + 1][upper_left.y + 2],                   \
    gradient_yy[upper_left.x + 1][upper_left.y + 2]) * 0.008207832296747465f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 1][upper_left.y + 3],                   \
    gradient_xy[upper_left.x + 1][upper_left.y + 3],                   \
    gradient_yy[upper_left.x + 1][upper_left.y + 3]) * 0.011942325524393555f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 1][upper_left.y + 4],                   \
    gradient_xy[upper_left.x + 1][upper_left.y + 4],                   \
    gradient_yy[upper_left.x + 1][upper_left.y + 4]) * 0.013532427693987032f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 1][upper_left.y + 5],                   \
    gradient_xy[upper_left.x + 1][upper_left.y + 5],                   \
    gradient_yy[upper_left.x + 1][upper_left.y + 5]) * 0.011942325524393555f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 1][upper_left.y + 6],                   \
    gradient_xy[upper_left.x + 1][upper_left.y + 6],                   \
    gradient_yy[upper_left.x + 1][upper_left.y + 6]) * 0.008207832296747465f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 1][upper_left.y + 7],                   \
    gradient_xy[upper_left.x + 1][upper_left.y + 7],                   \
    gradient_yy[upper_left.x + 1][upper_left.y + 7]) * 0.004393336040201352f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 1][upper_left.y + 8],                   \
    gradient_xy[upper_left.x + 1][upper_left.y + 8],                   \
    gradient_yy[upper_left.x + 1][upper_left.y + 8]) * 0.0018314149348447166f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 2][upper_left.y + 0],                   \
    gradient_xy[upper_left.x + 2][upper_left.y + 0],                   \
    gradient_yy[upper_left.x + 2][upper_left.y + 0]) * 0.0034215335484046386f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 2][upper_left.y + 1],                   \
    gradient_xy[upper_left.x + 2][upper_left.y + 1],                   \
    gradient_yy[upper_left.x + 2][upper_left.y + 1]) * 0.008207832296747465f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 2][upper_left.y + 2],                   \
    gradient_xy[upper_left.x + 2][upper_left.y + 2],                   \
    gradient_yy[upper_left.x + 2][upper_left.y + 2]) * 0.015334249507680085f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 2][upper_left.y + 3],                   \
    gradient_xy[upper_left.x + 2][upper_left.y + 3],                   \
    gradient_yy[upper_left.x + 2][upper_left.y + 3]) * 0.022311201383287904f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 2][upper_left.y + 4],                   \
    gradient_xy[upper_left.x + 2][upper_left.y + 4],                   \
    gradient_yy[upper_left.x + 2][upper_left.y + 4]) * 0.025281903333535125f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 2][upper_left.y + 5],                   \
    gradient_xy[upper_left.x + 2][upper_left.y + 5],                   \
    gradient_yy[upper_left.x + 2][upper_left.y + 5]) * 0.022311201383287904f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 2][upper_left.y + 6],                   \
    gradient_xy[upper_left.x + 2][upper_left.y + 6],                   \
    gradient_yy[upper_left.x + 2][upper_left.y + 6]) * 0.015334249507680085f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 2][upper_left.y + 7],                   \
    gradient_xy[upper_left.x + 2][upper_left.y + 7],                   \
    gradient_yy[upper_left.x + 2][upper_left.y + 7]) * 0.008207832296747465f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 2][upper_left.y + 8],                   \
    gradient_xy[upper_left.x + 2][upper_left.y + 8],                   \
    gradient_yy[upper_left.x + 2][upper_left.y + 8]) * 0.0034215335484046386f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 3][upper_left.y + 0],                   \
    gradient_xy[upper_left.x + 3][upper_left.y + 0],                   \
    gradient_yy[upper_left.x + 3][upper_left.y + 0]) * 0.004978301937756899f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 3][upper_left.y + 1],                   \
    gradient_xy[upper_left.x + 3][upper_left.y + 1],                   \
    gradient_yy[upper_left.x + 3][upper_left.y + 1]) * 0.011942325524393555f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 3][upper_left.y + 2],                   \
    gradient_xy[upper_left.x + 3][upper_left.y + 2],                   \
    gradient_yy[upper_left.x + 3][upper_left.y + 2]) * 0.022311201383287904f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 3][upper_left.y + 3],                   \
    gradient_xy[upper_left.x + 3][upper_left.y + 3],                   \
    gradient_yy[upper_left.x + 3][upper_left.y + 3]) * 0.03246260646250164f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 3][upper_left.y + 4],                   \
    gradient_xy[upper_left.x + 3][upper_left.y + 4],                   \
    gradient_yy[upper_left.x + 3][upper_left.y + 4]) * 0.03678495229550089f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 3][upper_left.y + 5],                   \
    gradient_xy[upper_left.x + 3][upper_left.y + 5],                   \
    gradient_yy[upper_left.x + 3][upper_left.y + 5]) * 0.03246260646250164f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 3][upper_left.y + 6],                   \
    gradient_xy[upper_left.x + 3][upper_left.y + 6],                   \
    gradient_yy[upper_left.x + 3][upper_left.y + 6]) * 0.022311201383287904f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 3][upper_left.y + 7],                   \
    gradient_xy[upper_left.x + 3][upper_left.y + 7],                   \
    gradient_yy[upper_left.x + 3][upper_left.y + 7]) * 0.011942325524393555f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 3][upper_left.y + 8],                   \
    gradient_xy[upper_left.x + 3][upper_left.y + 8],                   \
    gradient_yy[upper_left.x + 3][upper_left.y + 8]) * 0.004978301937756899f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 4][upper_left.y + 0],                   \
    gradient_xy[upper_left.x + 4][upper_left.y + 0],                   \
    gradient_yy[upper_left.x + 4][upper_left.y + 0]) * 0.005641155139668815f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 4][upper_left.y + 1],                   \
    gradient_xy[upper_left.x + 4][upper_left.y + 1],                   \
    gradient_yy[upper_left.x + 4][upper_left.y + 1]) * 0.013532427693987032f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 4][upper_left.y + 2],                   \
    gradient_xy[upper_left.x + 4][upper_left.y + 2],                   \
    gradient_yy[upper_left.x + 4][upper_left.y + 2]) * 0.025281903333535125f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 4][upper_left.y + 3],                   \
    gradient_xy[upper_left.x + 4][upper_left.y + 3],                   \
    gradient_yy[upper_left.x + 4][upper_left.y + 3]) * 0.03678495229550089f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 4][upper_left.y + 4],                   \
    gradient_xy[upper_left.x + 4][upper_left.y + 4],                   \
    gradient_yy[upper_left.x + 4][upper_left.y + 4]) * 0.041682811789783836f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 4][upper_left.y + 5],                   \
    gradient_xy[upper_left.x + 4][upper_left.y + 5],                   \
    gradient_yy[upper_left.x + 4][upper_left.y + 5]) * 0.03678495229550089f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 4][upper_left.y + 6],                   \
    gradient_xy[upper_left.x + 4][upper_left.y + 6],                   \
    gradient_yy[upper_left.x + 4][upper_left.y + 6]) * 0.025281903333535125f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 4][upper_left.y + 7],                   \
    gradient_xy[upper_left.x + 4][upper_left.y + 7],                   \
    gradient_yy[upper_left.x + 4][upper_left.y + 7]) * 0.013532427693987032f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 4][upper_left.y + 8],                   \
    gradient_xy[upper_left.x + 4][upper_left.y + 8],                   \
    gradient_yy[upper_left.x + 4][upper_left.y + 8]) * 0.005641155139668815f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 5][upper_left.y + 0],                   \
    gradient_xy[upper_left.x + 5][upper_left.y + 0],                   \
    gradient_yy[upper_left.x + 5][upper_left.y + 0]) * 0.004978301937756899f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 5][upper_left.y + 1],                   \
    gradient_xy[upper_left.x + 5][upper_left.y + 1],                   \
    gradient_yy[upper_left.x + 5][upper_left.y + 1]) * 0.011942325524393555f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 5][upper_left.y + 2],                   \
    gradient_xy[upper_left.x + 5][upper_left.y + 2],                   \
    gradient_yy[upper_left.x + 5][upper_left.y + 2]) * 0.022311201383287904f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 5][upper_left.y + 3],                   \
    gradient_xy[upper_left.x + 5][upper_left.y + 3],                   \
    gradient_yy[upper_left.x + 5][upper_left.y + 3]) * 0.03246260646250164f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 5][upper_left.y + 4],                   \
    gradient_xy[upper_left.x + 5][upper_left.y + 4],                   \
    gradient_yy[upper_left.x + 5][upper_left.y + 4]) * 0.03678495229550089f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 5][upper_left.y + 5],                   \
    gradient_xy[upper_left.x + 5][upper_left.y + 5],                   \
    gradient_yy[upper_left.x + 5][upper_left.y + 5]) * 0.03246260646250164f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 5][upper_left.y + 6],                   \
    gradient_xy[upper_left.x + 5][upper_left.y + 6],                   \
    gradient_yy[upper_left.x + 5][upper_left.y + 6]) * 0.022311201383287904f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 5][upper_left.y + 7],                   \
    gradient_xy[upper_left.x + 5][upper_left.y + 7],                   \
    gradient_yy[upper_left.x + 5][upper_left.y + 7]) * 0.011942325524393555f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 5][upper_left.y + 8],                   \
    gradient_xy[upper_left.x + 5][upper_left.y + 8],                   \
    gradient_yy[upper_left.x + 5][upper_left.y + 8]) * 0.004978301937756899f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 6][upper_left.y + 0],                   \
    gradient_xy[upper_left.x + 6][upper_left.y + 0],                   \
    gradient_yy[upper_left.x + 6][upper_left.y + 0]) * 0.0034215335484046386f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 6][upper_left.y + 1],                   \
    gradient_xy[upper_left.x + 6][upper_left.y + 1],                   \
    gradient_yy[upper_left.x + 6][upper_left.y + 1]) * 0.008207832296747465f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 6][upper_left.y + 2],                   \
    gradient_xy[upper_left.x + 6][upper_left.y + 2],                   \
    gradient_yy[upper_left.x + 6][upper_left.y + 2]) * 0.015334249507680085f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 6][upper_left.y + 3],                   \
    gradient_xy[upper_left.x + 6][upper_left.y + 3],                   \
    gradient_yy[upper_left.x + 6][upper_left.y + 3]) * 0.022311201383287904f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 6][upper_left.y + 4],                   \
    gradient_xy[upper_left.x + 6][upper_left.y + 4],                   \
    gradient_yy[upper_left.x + 6][upper_left.y + 4]) * 0.025281903333535125f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 6][upper_left.y + 5],                   \
    gradient_xy[upper_left.x + 6][upper_left.y + 5],                   \
    gradient_yy[upper_left.x + 6][upper_left.y + 5]) * 0.022311201383287904f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 6][upper_left.y + 6],                   \
    gradient_xy[upper_left.x + 6][upper_left.y + 6],                   \
    gradient_yy[upper_left.x + 6][upper_left.y + 6]) * 0.015334249507680085f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 6][upper_left.y + 7],                   \
    gradient_xy[upper_left.x + 6][upper_left.y + 7],                   \
    gradient_yy[upper_left.x + 6][upper_left.y + 7]) * 0.008207832296747465f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 6][upper_left.y + 8],                   \
    gradient_xy[upper_left.x + 6][upper_left.y + 8],                   \
    gradient_yy[upper_left.x + 6][upper_left.y + 8]) * 0.0034215335484046386f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 7][upper_left.y + 0],                   \
    gradient_xy[upper_left.x + 7][upper_left.y + 0],                   \
    gradient_yy[upper_left.x + 7][upper_left.y + 0]) * 0.0018314149348447166f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 7][upper_left.y + 1],                   \
    gradient_xy[upper_left.x + 7][upper_left.y + 1],                   \
    gradient_yy[upper_left.x + 7][upper_left.y + 1]) * 0.004393336040201352f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 7][upper_left.y + 2],                   \
    gradient_xy[upper_left.x + 7][upper_left.y + 2],                   \
    gradient_yy[upper_left.x + 7][upper_left.y + 2]) * 0.008207832296747465f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 7][upper_left.y + 3],                   \
    gradient_xy[upper_left.x + 7][upper_left.y + 3],                   \
    gradient_yy[upper_left.x + 7][upper_left.y + 3]) * 0.011942325524393555f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 7][upper_left.y + 4],                   \
    gradient_xy[upper_left.x + 7][upper_left.y + 4],                   \
    gradient_yy[upper_left.x + 7][upper_left.y + 4]) * 0.013532427693987032f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 7][upper_left.y + 5],                   \
    gradient_xy[upper_left.x + 7][upper_left.y + 5],                   \
    gradient_yy[upper_left.x + 7][upper_left.y + 5]) * 0.011942325524393555f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 7][upper_left.y + 6],                   \
    gradient_xy[upper_left.x + 7][upper_left.y + 6],                   \
    gradient_yy[upper_left.x + 7][upper_left.y + 6]) * 0.008207832296747465f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 7][upper_left.y + 7],                   \
    gradient_xy[upper_left.x + 7][upper_left.y + 7],                   \
    gradient_yy[upper_left.x + 7][upper_left.y + 7]) * 0.004393336040201352f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 7][upper_left.y + 8],                   \
    gradient_xy[upper_left.x + 7][upper_left.y + 8],                   \
    gradient_yy[upper_left.x + 7][upper_left.y + 8]) * 0.0018314149348447166f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 8][upper_left.y + 0],                   \
    gradient_xy[upper_left.x + 8][upper_left.y + 0],                   \
    gradient_yy[upper_left.x + 8][upper_left.y + 0]) * 0.0007634473286087523f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 8][upper_left.y + 1],                   \
    gradient_xy[upper_left.x + 8][upper_left.y + 1],                   \
    gradient_yy[upper_left.x + 8][upper_left.y + 1]) * 0.0018314149348447166f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 8][upper_left.y + 2],                   \
    gradient_xy[upper_left.x + 8][upper_left.y + 2],                   \
    gradient_yy[upper_left.x + 8][upper_left.y + 2]) * 0.0034215335484046386f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 8][upper_left.y + 3],                   \
    gradient_xy[upper_left.x + 8][upper_left.y + 3],                   \
    gradient_yy[upper_left.x + 8][upper_left.y + 3]) * 0.004978301937756899f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 8][upper_left.y + 4],                   \
    gradient_xy[upper_left.x + 8][upper_left.y + 4],                   \
    gradient_yy[upper_left.x + 8][upper_left.y + 4]) * 0.005641155139668815f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 8][upper_left.y + 5],                   \
    gradient_xy[upper_left.x + 8][upper_left.y + 5],                   \
    gradient_yy[upper_left.x + 8][upper_left.y + 5]) * 0.004978301937756899f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 8][upper_left.y + 6],                   \
    gradient_xy[upper_left.x + 8][upper_left.y + 6],                   \
    gradient_yy[upper_left.x + 8][upper_left.y + 6]) * 0.0034215335484046386f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 8][upper_left.y + 7],                   \
    gradient_xy[upper_left.x + 8][upper_left.y + 7],                   \
    gradient_yy[upper_left.x + 8][upper_left.y + 7]) * 0.0018314149348447166f;   \
undecomposed.xyw += vec3(                                                  \
    gradient_xx[upper_left.x + 8][upper_left.y + 8],                   \
    gradient_xy[upper_left.x + 8][upper_left.y + 8],                   \
    gradient_yy[upper_left.x + 8][upper_left.y + 8]) * 0.0007634473286087523f;  
