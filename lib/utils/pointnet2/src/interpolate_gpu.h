#ifndef _INTERPOLATE_GPU_H
#define _INTERPOLATE_GPU_H
#include "paddle/extension.h"
#include <vector>


std::vector<paddle::Tensor> three_nn_wrapper_fast(const paddle::Tensor& unknown_tensor, 
   const paddle::Tensor& known_tensor,  const paddle::Tensor& dist2_tensor,  const paddle::Tensor& idx_tensor,
   int b, int n, int m);


void three_nn_kernel_launcher_fast(int b, int n, int m,  float *unknown,
	float *known, float *dist2, int *idx);


std::vector<paddle::Tensor> three_interpolate_wrapper_fast(const paddle::Tensor& points_tensor, 
     const paddle::Tensor& idx_tensor,  const paddle::Tensor& weight_tensor,  const paddle::Tensor& out_tensor,int b, int c, int m, int n);

void three_interpolate_kernel_launcher_fast(int b, int c, int m, int n, 
    float *points, int *idx, float *weight, float *out);


std::vector<paddle::Tensor> three_interpolate_grad_wrapper_fast(const paddle::Tensor& grad_out_tensor, 
    const  paddle::Tensor& idx_tensor,  const paddle::Tensor& weight_tensor,  const paddle::Tensor& grad_points_tensor,int b, int c, int n, int m);

void three_interpolate_grad_kernel_launcher_fast(int b, int c, int n, int m, float *grad_out, 
    int *idx, float *weight, float *grad_points);

#endif
