#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H

#include "paddle/extension.h"
#include <vector>

std::vector<paddle::Tensor> ball_query_wrapper_fast(const paddle::Tensor& new_xyz_tensor, const paddle::Tensor& xyz_tensor, const paddle::Tensor& idx_tensor,
int b, int n, int m, float radius, int nsample);

void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, 
	float *xyz, float *new_xyz, int *idx);

#endif