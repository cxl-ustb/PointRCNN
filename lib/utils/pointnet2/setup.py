from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='pointnet2',
    ext_modules=CUDAExtension(
        sources=['./src/sampling_gpu.cu',
        './src/sampling.cpp',
        './src/pointnet2_api.cpp',
        './src/ball_query_gpu.cu',
        './src/ball_query.cpp',
        './src/group_points_gpu.cu',
        './src/group_points.cpp',
        './src/interpolate_gpu.cu',
        './src/interpolate.cpp']
    )
)
