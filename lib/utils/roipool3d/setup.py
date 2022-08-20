from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='roipool3d',
    ext_modules=CUDAExtension(
        sources=['./src/roipool3d_kernel.cu','./src/roipool3d.cpp']
    )
)
