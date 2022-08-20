#include "paddle/extension.h"
#include <vector>
#include "ball_query_gpu.h"

#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(),#x" must be a GPU Tensor.")
#define CHECK_INPUT(x) CHECK_CUDA(x)

std::vector<paddle::Tensor>  ball_query_wrapper_fast(const paddle::Tensor& new_xyz_tensor,
 const paddle::Tensor& xyz_tensor, const paddle::Tensor& idx_tensor,
 int b, int n, int m, float radius, int nsample) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    float *new_xyz = const_cast<float*>(new_xyz_tensor.data<float>());
    float *xyz = const_cast<float*>(xyz_tensor.data<float>());
    int *idx =const_cast<int*>(idx_tensor.data<int>());
    ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx);
    return {idx_tensor};
}


