#ifndef _GROUP_POINTS_GPU_H
#define _GROUP_POINTS_GPU_H
#include"cuda_utils.h"
#include "paddle/extension.h"
#include <vector>


std::vector<paddle::Tensor> group_points_wrapper_fast(
const paddle::Tensor& points_tensor, 
const paddle::Tensor& idx_tensor,const  paddle::Tensor& out_tensor,
int b, int c, int n, int npoints, int nsample);

void group_points_kernel_launcher_fast(int b, int c, int n, 
int npoints, int nsample, 
    float *points, int *idx, float *out);

std::vector<paddle::Tensor>  group_points_grad_wrapper_fast(
    const paddle::Tensor& grad_out_tensor, const paddle::Tensor& idx_tensor, 
    const paddle::Tensor& grad_points_tensor,
    int b, int c, int n, int npoints, int nsample);

void group_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, 
    int nsample, 
    float *grad_out, int *idx, float *grad_points);

#endif
