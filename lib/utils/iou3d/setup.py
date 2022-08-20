from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='iou3d',
    ext_modules=CUDAExtension(
        sources=['./src/iou3d_kernel.cu','./src/iou3d.cpp']
    )
)
