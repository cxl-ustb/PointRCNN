#include "paddle/extension.h"
#include <vector>
#include <stdlib.h>
#include <glog/logging.h>
#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(),#x" must be a GPU Tensor.")
#define CHECK_INPUT(x) CHECK_CUDA(x)


std::vector<paddle::Tensor> roipool3dLauncher(int batch_size, int pts_num, int boxes_num, 
				int feature_in_len, int sampled_pts_num, 
                       float *xyz, float *boxes3d, float *pts_feature, 	
					   int* pooled_empty_flag);


std::vector<paddle::Tensor> roipool3d_gpu(const paddle::Tensor& xyz, 
const paddle::Tensor& boxes3d,
 const paddle::Tensor& pts_feature, 
 const paddle::Tensor&  pooled_features, 
  const paddle::Tensor& pooled_empty_flag){
    // params xyz: (B, N, 3)
    // params boxes3d: (B, M, 7)
    // params pts_feature: (B, N, C)
    // params pooled_features: (B, M, 512, 3+C)
    // params pooled_empty_flag: (B, M)
    CHECK_INPUT(xyz);
    CHECK_INPUT(boxes3d);
    CHECK_INPUT(pts_feature);
    CHECK_INPUT(pooled_features);
    CHECK_INPUT(pooled_empty_flag);
   
    int batch_size = xyz.shape()[0];
    int pts_num = xyz.shape()[1];
    int boxes_num = boxes3d.shape()[1];
    int feature_in_len = pts_feature.shape()[2];
    int sampled_pts_num = pooled_features.shape()[2];


    float* xyz_data = const_cast < float*>(xyz.data<float>());
    float * boxes3d_data = const_cast < float*>(boxes3d.data<float>());
    float * pts_feature_data = const_cast < float*>(pts_feature.data<float>());
    int * pooled_empty_flag_data = const_cast <int*>(pooled_empty_flag.data<int>());

    return roipool3dLauncher(batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num, 
                       xyz_data, boxes3d_data, pts_feature_data,pooled_empty_flag_data);
    
    

}

std::vector<std::vector<int64_t>> roipool3d_gpu_infershape(std::vector<int64_t> xyz_shape,
                                                                                                    std::vector<int64_t> boxes3d_shape,
                                                                                                    std::vector<int64_t> pts_feature_shape,
                                                                                                    std::vector<int64_t> pooled_features_shape,
                                                                                                    std::vector<int64_t> pooled_empty_flag_shape){
                                                                                                        return {{xyz_shape[0],boxes3d_shape[0],pooled_empty_flag_shape[1],3+pts_feature_shape[2]}};
                                                                                                    }

std::vector<paddle::DataType> roipool3d_gpu_inferdtype(paddle::DataType xyz_dtype,
                                                                                                    paddle::DataType boxes3d_dtype,
                                                                                                    paddle::DataType pts_feature_dtype,
                                                                                                    paddle::DataType pooled_features_dtype,
                                                                                                    paddle::DataType pooled_empty_flag_dtype){
                                                                                                        return {xyz_dtype};
                                                                                                    }
PD_BUILD_OP(forward)
    .Inputs({"XYZ", "BOXES3D","PTS_FEATURE","POOLED_FEATURES","POOLED_EMPTY_FLAG"})
    .Outputs({"Output"})
    .SetKernelFn(PD_KERNEL(roipool3d_gpu))
    .SetInferShapeFn(PD_INFER_SHAPE(roipool3d_gpu_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(roipool3d_gpu_inferdtype));

int pt_in_box3d_cpu(float x, float y, float z, float cx, float bottom_y,
 float cz, float h, float w, float l, float angle){
    float max_dis = 10.0, x_rot, z_rot, cosa, sina, cy;
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


std::vector<paddle::Tensor> pts_in_boxes3d_cpu(
    const paddle::Tensor& pts_flag,
     const paddle::Tensor& pts,
      const paddle::Tensor& boxes3d){
    // param pts_flag: (M, N), 0 or 1
    // param pts: (N, 3)
    // param boxes3d: (M, 7)  [x, y, z, h, w, l, ry]

    long boxes_num = boxes3d.shape()[0];
    long pts_num = pts.shape()[0];

    long * pts_flag_flat = const_cast < long*>(pts_flag.data<long>());
    float * pts_flat = const_cast < float*>(pts.data<float>());
    float * boxes3d_flat = const_cast < float*>(boxes3d.data<float>());

    memset(pts_flag_flat, 0, boxes_num * pts_num * sizeof(long));

    int i, j, cur_in_flag;
    for (i = 0; i < boxes_num; i++){
        for (j = 0; j < pts_num; j++){
            cur_in_flag = pt_in_box3d_cpu(pts_flat[j * 3], pts_flat[j * 3 + 1], pts_flat[j * 3 + 2], boxes3d_flat[i * 7],
                                          boxes3d_flat[i * 7 + 1], boxes3d_flat[i * 7 + 2], boxes3d_flat[i * 7 + 3],
                                          boxes3d_flat[i * 7 + 4], boxes3d_flat[i * 7 + 5], boxes3d_flat[i * 7 + 6]);
            pts_flag_flat[i * pts_num + j] = cur_in_flag;
        }
    }
    return {pts_flag};
}

std::vector<std::vector<int64_t>> pts_in_boxes3d_cpu_infershape(std::vector<int64_t> pts_flag_shape,
                                                                                                    std::vector<int64_t> pts_shape,
                                                                                                    std::vector<int64_t> boxes3d_shape){
                                                                                                        return {{pts_flag_shape[0],pts_flag_shape[1]}};
                                                                                                    }

std::vector<paddle::DataType> pts_in_boxes3d_cpu_inferdtype(paddle::DataType pts_flag_dtype,
                                                                                                    paddle::DataType pts_dtype,
                                                                                                    paddle::DataType boxes3d_dtype){
                                                                                                        return {pts_flag_dtype};
                                                                                                    }
//paddle::Tensor pts_flag, paddle::Tensor pts, paddle::Tensor boxes3d
PD_BUILD_OP(pts_in_boxes3d_cpu)
    .Inputs({"PTS_FLAG", "PTS","BOXES3D"})
    .Outputs({"Output"})
    .SetKernelFn(PD_KERNEL(pts_in_boxes3d_cpu))
    .SetInferShapeFn(PD_INFER_SHAPE(pts_in_boxes3d_cpu_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(pts_in_boxes3d_cpu_inferdtype));


std::vector<paddle::Tensor> roipool3d_cpu(const paddle::Tensor& pts, const paddle::Tensor& boxes3d, 
const paddle::Tensor& pts_feature, const paddle::Tensor& pooled_pts,
                  const paddle::Tensor& pooled_features, const paddle::Tensor& pooled_empty_flag){
    // param pts: (N, 3) [x, y, z]
    // param boxes3d: (M, 7) [x, y, z, h, w, l, ry]
    // param pts_feature: (N, C)
    // param pooled_pts: (M, 512, 3)
    // param pooled_features: (M, 512, C)
    int boxes_num = boxes3d.shape()[0];
    int pts_num = pts.shape()[0];
    int feature_len = pts_feature.shape()[1];
    int sampled_pts_num = pooled_pts.shape()[1];
    
    float * pts_flat = const_cast < float*>(pts.data<float>());
    float * boxes3d_flat = const_cast < float*>(boxes3d.data<float>());
    float * pts_feature_flat = const_cast < float*>(pts_feature.data<float>());
    float * pooled_pts_flat = const_cast < float*>(pooled_pts.data<float>());
    float * pooled_features_flat = const_cast < float*>(pooled_features.data<float>());
    int * pooled_empty_flag_flat = const_cast < int*>(pooled_empty_flag.data<int>());

    memset(pooled_empty_flag_flat, 0, boxes_num * sizeof(int));

    int i, j, k, cnt, temp_idx, duplicate_idx, cur_in_flag;
    for (i = 0; i < boxes_num; i++){
        cnt = 0;
        for (j = 0; j < pts_num; j++){
            cur_in_flag = pt_in_box3d_cpu(pts_flat[j * 3], pts_flat[j * 3 + 1], pts_flat[j * 3 + 2], boxes3d_flat[i * 7],
                                       boxes3d_flat[i * 7 + 1], boxes3d_flat[i * 7 + 2], boxes3d_flat[i * 7 + 3],
                                       boxes3d_flat[i * 7 + 4], boxes3d_flat[i * 7 + 5], boxes3d_flat[i * 7 + 6]);

            if (cur_in_flag){
                if (cnt < sampled_pts_num){
                    temp_idx = i * sampled_pts_num * 3 + cnt * 3;
                    for (k = 0; k < 3; k++) pooled_pts_flat[temp_idx + k] = pts_flat[j * 3 + k];
                    temp_idx = i * sampled_pts_num * feature_len + cnt * feature_len;
                    for (k = 0; k < feature_len; k++) pooled_features_flat[temp_idx + k] = pts_feature_flat[j * feature_len + k];
                    cnt++;
                }
                else break;
            }
        }

        if (cnt == 0){
            // no points in this box
            pooled_empty_flag_flat[i] = 1;
        }
        else if (cnt < sampled_pts_num){
            // duplicate same points
            duplicate_idx = 0;
            LOG(INFO) << "info test";  
            for (j = cnt; j < sampled_pts_num; j++){
                temp_idx = i * sampled_pts_num * 3 + j * 3;
                duplicate_idx = i * sampled_pts_num * 3 + (j % cnt) * 3;
                for (k = 0; k < 3; k++) pooled_pts_flat[temp_idx + k] = pooled_pts_flat[duplicate_idx + k];
                temp_idx = i * sampled_pts_num * feature_len + j * feature_len;
                duplicate_idx = i * sampled_pts_num * feature_len + (j % cnt) * feature_len;
                for (k = 0; k < feature_len; k++){
                    pooled_features_flat[temp_idx + k] = pooled_features_flat[duplicate_idx + k];
                }
            }
        }
    }
  
    return {pooled_features};
}

std::vector<std::vector<int64_t>> roipool3d_cpu_infershape(std::vector<int64_t> pts_shape,
                                                                                                    std::vector<int64_t> boxes3d_shape,
                                                                                                    std::vector<int64_t> pts_feature_shape,
                                                                                                    std::vector<int64_t> pooled_pts_shape,
                                                                                                    std::vector<int64_t> pooled_features_shape,
                                                                                                    std::vector<int64_t> pooled_empty_flag_shape){
                                                                                                        return {{pooled_features_shape[0],pooled_features_shape[1],pooled_features_shape[2]}};
                                                                                                    }

std::vector<paddle::DataType> roipool3d_cpu_inferdtype(paddle::DataType pts_dtype,
                                                                                                    paddle::DataType boxes3d_dtype,
                                                                                                    paddle::DataType pts_feature_dtype,
                                                                                                    paddle::DataType pooled_pts_dtype,
                                                                                                    paddle::DataType pooled_features_dtype,
                                                                                                    paddle::DataType  pooled_empty_flag_dtype){
                                                                                                        return {pts_dtype};
                                                                                                    }

PD_BUILD_OP(roipool3d_cpu)
    .Inputs({"PTS","BOXES3D","PTS_FEATURE","POOLED_PTS","POOLED_FEATURES","POOLED_EMPTY_FLAG"})
    .Outputs({"Output"})
    .SetKernelFn(PD_KERNEL(roipool3d_cpu))
    .SetInferShapeFn(PD_INFER_SHAPE(roipool3d_cpu_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(roipool3d_cpu_inferdtype));
