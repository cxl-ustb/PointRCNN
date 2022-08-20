#include "sampling_gpu.h"
#include"paddle/extension.h"


std::vector<paddle::Tensor> gather_points_wrapper_fast(
    const paddle::Tensor& points_tensor, const paddle::Tensor& idx_tensor, 
    const paddle::Tensor& out_tensor,int b, int c, int n, int npoints){
   
    float *points = const_cast<float*>(points_tensor.data<float>());
    int *idx =const_cast<int*>(idx_tensor.data<int>());
    float *out = const_cast<float*>(out_tensor.data<float>());

    gather_points_kernel_launcher_fast(b, c, n, npoints, points, idx, out);
    
    return {out_tensor};
}

std::vector<paddle::Tensor> gather_points_grad_wrapper_fast( 
    const paddle::Tensor& grad_out_tensor, const paddle::Tensor& idx_tensor, 
    const paddle::Tensor& grad_points_tensor,int b, int c, int n, int npoints) {

    float *grad_out =  const_cast<float*>(grad_out_tensor.data<float>());
    int *idx =const_cast<int*>(idx_tensor.data<int>());
    float *grad_points =const_cast<float*>(grad_points_tensor.data<float>());

    gather_points_grad_kernel_launcher_fast(b, c, n, npoints, grad_out, idx, grad_points);
    return {grad_points_tensor};
}

std::vector<paddle::Tensor> furthest_point_sampling_wrapper(
    const paddle::Tensor& points_tensor, const paddle::Tensor& temp_tensor,
     const paddle::Tensor& idx_tensor,int b, int n, int m) {

    float *points = const_cast<float*>(points_tensor.data<float>());
    float *temp = const_cast<float*>(temp_tensor.data<float>());
    int *idx = const_cast<int*>(idx_tensor.data<int>());

    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);
    return {idx_tensor};
}

