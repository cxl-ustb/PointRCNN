#include "paddle/extension.h"
#include <vector>
#include "group_points_gpu.h"

#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(),#x" must be a GPU Tensor.")
#define CHECK_INPUT(x) CHECK_CUDA(x)


std::vector<paddle::Tensor> group_points_wrapper_fast(const paddle::Tensor& points_tensor, 
const paddle::Tensor& idx_tensor,const  paddle::Tensor& out_tensor,
int b, int c, int n, int npoints, int nsample){

    CHECK_INPUT(points_tensor);
    CHECK_INPUT(idx_tensor);
    float *points = const_cast<float*>(points_tensor.data<float>());
    int *idx = const_cast<int*>(idx_tensor.data<int>());
    float *out = const_cast<float*>(out_tensor.data<float>());
   
    group_points_kernel_launcher_fast(b, c, n, npoints, nsample, points, idx, out);
    return {out_tensor};
}




std::vector<paddle::Tensor>  group_points_grad_wrapper_fast(
    const paddle::Tensor& grad_out_tensor, 
    const paddle::Tensor& idx_tensor, 
    const paddle::Tensor& grad_points_tensor,
    int b, int c, int n, int npoints, int nsample) {
    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    float *grad_points = const_cast<float*>(grad_points_tensor.data<float>());
    int *idx = const_cast<int*>(idx_tensor.data<int>());
    float *grad_out = const_cast<float*>(grad_out_tensor.data<float>());

    group_points_grad_kernel_launcher_fast(b, c, n, npoints, nsample, 
    grad_out, idx, grad_points);
    return {grad_points_tensor};
}

