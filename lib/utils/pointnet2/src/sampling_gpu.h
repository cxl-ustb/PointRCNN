#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

#include "paddle/extension.h"
#include<vector>
std::vector<paddle::Tensor> gather_points_wrapper_fast(
    const paddle::Tensor& points_tensor, const paddle::Tensor& idx_tensor, 
    const paddle::Tensor& out_tensor,int b, int c, int n, int npoints);

void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints, 
 float *points, int *idx, float *out);


std::vector<paddle::Tensor> gather_points_grad_wrapper_fast( 
    const paddle::Tensor& grad_out_tensor, const paddle::Tensor& idx_tensor, 
    const paddle::Tensor& grad_points_tensor,int b, int c, int n, int npoints);

void gather_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, 
    float *grad_out, int *idx, float *grad_points);


std::vector<paddle::Tensor> furthest_point_sampling_wrapper(
    const paddle::Tensor& points_tensor, const paddle::Tensor& temp_tensor,
     const paddle::Tensor& idx_tensor,int b, int n, int m);

void furthest_point_sampling_kernel_launcher(int b, int n, int m, 
    float *dataset, float *temp, int *idxs);

#endif