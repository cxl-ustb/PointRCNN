#include <math.h>
#include <stdio.h>
#include "paddle/extension.h"

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

template<typename T>
__device__ inline int pt_in_box3d(T x, T y, T z, T cx, T bottom_y, T cz, T h, T w,
                              T l, T angle, T max_dis){
    T x_rot, z_rot, cosa, sina, cy;
    int in_flag;
    cy = bottom_y - h / 2.0;
    if ((fabsf(x - cx) > max_dis) || (fabsf(y - cy) > h / 2.0) || (fabsf(z - cz) > max_dis)){
        return 0;
    }
    cosa = cos(angle); sina = sin(angle);
    x_rot = (x - cx) * cosa + (z - cz) * (-sina);
    z_rot = (x - cx) * sina + (z - cz) * cosa;

    in_flag = (x_rot >= -l / 2.0) & (x_rot <= l / 2.0) & (z_rot >= -w / 2.0) & (z_rot <= w / 2.0);
    return in_flag;
}

// 对于box内的点进行采样
template<typename T>
__global__ void roipool3d_forward(int batch_size, int pts_num, int boxes_num,
								  int feature_in_len, int sampled_pts_num, 
                                  const T *xyz, const T *boxes3d, const T  *pts_feature, 
                                  T  *pooled_features, int* pooled_empty_flag){
    // params xyz: (B, N, 3)
    // params boxes3d: (B, M, 7)
    // params pts_feature: (B, N, C)
    // params pooled_features: (B, M, 512, 3+C)
    // params pooled_empty_flag: (B, M)

    int boxes_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boxes_idx >= boxes_num){
        return;
    }
    // pts_num=N,boxes_num=M
    for (int i = 0; i < batch_size; i++){
        int cnt = 0;
        for (int k = 0; k < pts_num; k++){
            int pt_offset = i * pts_num * 3 + k * 3;
            int box_offset = i * boxes_num * 7 + boxes_idx * 7;

            int cur_in_flag = pt_in_box3d(xyz[pt_offset], xyz[pt_offset + 1], xyz[pt_offset + 2], boxes3d[box_offset], 
                                          boxes3d[box_offset + 1], boxes3d[box_offset + 2], boxes3d[box_offset + 3], 
                                          boxes3d[box_offset + 4], boxes3d[box_offset + 5], boxes3d[box_offset + 6], 10.0);
            if (cur_in_flag){
                if (cnt < sampled_pts_num){
                    // (B, M, 512, 3+C)
                    int feature_out_offset = i * boxes_num * sampled_pts_num * (3 + feature_in_len) + 
                                             boxes_idx * sampled_pts_num * (3 + feature_in_len) + 
                                             cnt * (3 + feature_in_len);

                    int feature_in_offset = i * pts_num * feature_in_len + k * feature_in_len;

                    // copy xyz
                    for (int j = 0; j < 3; j++)
                        pooled_features[feature_out_offset + j] = xyz[pt_offset + j];

                    // copy feature
                    for (int j = 0; j < feature_in_len; j++)
                        pooled_features[feature_out_offset + 3 + j] = pts_feature[feature_in_offset + j];

                    cnt++;
                }
                else break;
            }
        }

        if (cnt == 0){
            pooled_empty_flag[i * boxes_num + boxes_idx] = 1;
        }
        else if (cnt < sampled_pts_num){
            // duplicate same points for sampling
            for (int k = cnt; k < sampled_pts_num; k++){
                int duplicate_idx = k % cnt;
                int src_offset = i * boxes_num * sampled_pts_num * (3 + feature_in_len) + 
                                 boxes_idx * sampled_pts_num * (3 + feature_in_len) + 
                                 duplicate_idx * (3 + feature_in_len);
                int dst_offset = i * boxes_num * sampled_pts_num * (3 + feature_in_len) + 
                                 boxes_idx * sampled_pts_num * (3 + feature_in_len) + 
                                 k * (3 + feature_in_len);
                for (int j = 0; j < 3 + feature_in_len; j++)
                    pooled_features[dst_offset + j] = pooled_features[src_offset + j];
            }
        }
    }
}

template<typename T>
__global__ void assign_pts_to_box3d(int batch_size, int pts_num, int boxes_num, T *xyz, T *boxes3d, int* pts_assign){
    // params xyz: (B, N, 3)
    // params boxes3d: (B, M, 7)
    // params pts_assign: (B, N, M): idx of the corresponding box3d, -1 means background points
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = blockIdx.z;
     
    if (pt_idx >= pts_num || box_idx >= boxes_num || bs_idx >= batch_size){
        return;
    }
    int assign_idx = bs_idx * pts_num * boxes_num + pt_idx * boxes_num + box_idx;
    pts_assign[assign_idx] = 0;

    int box_offset = bs_idx * boxes_num * 7 + box_idx * 7;
    int pt_offset = bs_idx * pts_num * 3 + pt_idx * 3;
    float max_dis=10.0;
    int cur_in_flag = pt_in_box3d(xyz[pt_offset], xyz[pt_offset + 1], xyz[pt_offset + 2], boxes3d[box_offset], 
                                  boxes3d[box_offset + 1], boxes3d[box_offset + 2], boxes3d[box_offset + 3], 
                                  boxes3d[box_offset + 4], boxes3d[box_offset + 5], boxes3d[box_offset + 6], max_dis);

    pts_assign[assign_idx] = cur_in_flag;
}

template<typename T>
__global__ void get_pooled_idx(int batch_size, int pts_num, int boxes_num, int sampled_pts_num, 
                               const int* pts_assign, int* pts_idx, int* pooled_empty_flag){
    // params xyz: (B, N, 3)
    // params pts_feature: (B, N, C)
    // params pts_assign: (B, N, M): idx of the corresponding box3d, -1 means background points
    // params pts_idx: (B, M, 512)
    // params pooled_empty_flag: (B, M)

    int boxes_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (boxes_idx >= boxes_num){
        return;
    }

    int bs_idx = blockIdx.y;

    int cnt = 0;
    for (int k = 0; k < pts_num; k++){
        if (pts_assign[bs_idx * pts_num * boxes_num + k * boxes_num + boxes_idx]){
            if (cnt < sampled_pts_num){
                pts_idx[bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num + cnt] = k;
                cnt++;
            }
            else break;
        }
    }

    if (cnt == 0){
        pooled_empty_flag[bs_idx * boxes_num + boxes_idx] = 1;
    }
    else if (cnt < sampled_pts_num){
        // duplicate same points for sampling
        for (int k = cnt; k < sampled_pts_num; k++){
            int duplicate_idx = k % cnt;
            int base_offset = bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num;
            pts_idx[base_offset + k] = pts_idx[base_offset + duplicate_idx];
        }
    }
}

template<typename T>
__global__ void roipool3d_forward(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num, 
                                   T *xyz, int*pts_idx, T * pts_feature, 
                                   T * pooled_features, int* pooled_empty_flag){
    // params xyz: (B, N, 3)
    // params pts_idx: (B, M, 512)
    // params pts_feature: (B, N, C)
    // params pooled_features: (B, M, 512, 3+C)
    // params pooled_empty_flag: (B, M)
    
    int sample_pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;
    int bs_idx = blockIdx.z;
    
    if (sample_pt_idx >= sampled_pts_num || box_idx >= boxes_num || bs_idx >= batch_size){
        return;
    }

    if (pooled_empty_flag[bs_idx * boxes_num + box_idx]){
        return;
    }

    int temp_idx = bs_idx * boxes_num * sampled_pts_num + box_idx * sampled_pts_num + sample_pt_idx;
    int src_pt_idx = pts_idx[temp_idx];
    int dst_feature_offset = temp_idx * (3 + feature_in_len);

    for (int j = 0; j < 3; j++)
        pooled_features[dst_feature_offset + j] = xyz[bs_idx * pts_num * 3 + src_pt_idx * 3 + j];

    int src_feature_offset = bs_idx * pts_num * feature_in_len + src_pt_idx * feature_in_len;
    for (int j = 0; j < feature_in_len; j++)
        pooled_features[dst_feature_offset + 3 + j] = pts_feature[src_feature_offset + j];
}



std::vector<paddle::Tensor> roipool3dLauncher(int batch_size, int pts_num, int boxes_num, 
				int feature_in_len, int sampled_pts_num, 
                       float *xyz, float *boxes3d, float *pts_feature, 	int* pooled_empty_flag){
	
    auto pts_assign = paddle::Tensor(paddle::PlaceType::kGPU, {batch_size, pts_num, boxes_num});
	
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num, batch_size); 
    dim3 threads(THREADS_PER_BLOCK);

	PD_DISPATCH_FLOATING_TYPES(
		paddle::DataType::FLOAT32,
        "assign_pts_to_box3d",
        ([&] {
			assign_pts_to_box3d<float>
			<<<blocks, threads>>>
			(batch_size, pts_num, boxes_num, 
				xyz, 
				boxes3d,
				 pts_assign.mutable_data<int>());
        }));

	auto pts_idx = paddle::Tensor(paddle::PlaceType::kGPU, {batch_size, boxes_num, sampled_pts_num});

    dim3 blocks2(DIVUP(boxes_num, THREADS_PER_BLOCK), batch_size);  
 
	PD_DISPATCH_FLOATING_TYPES(
		paddle::DataType::FLOAT32,
        "get_pooled_idx",
        ([&] {
			get_pooled_idx<float>
			<<<blocks2, threads>>>
			(batch_size, pts_num, boxes_num, 
				sampled_pts_num,
				pts_assign.data<int>(),
				pts_idx.mutable_data<int>(),
				pooled_empty_flag);
        }));

	auto pooled_features = paddle::Tensor(paddle::PlaceType::kGPU, {batch_size, boxes_num, sampled_pts_num,3+feature_in_len});
    dim3 blocks_pool(DIVUP(sampled_pts_num, THREADS_PER_BLOCK), boxes_num, batch_size); 
   
	PD_DISPATCH_FLOATING_TYPES(
		paddle::DataType::FLOAT32,
        "roipool3d_forward",
        ([&] {
			roipool3d_forward<float>
			<<<blocks_pool, threads>>>
			(batch_size, pts_num, boxes_num, feature_in_len,
				sampled_pts_num,xyz,
				pts_idx.data<int>(),pts_feature,
				pooled_features.mutable_data<float>(),
				pooled_empty_flag);
        }));
	return {pooled_features};
}