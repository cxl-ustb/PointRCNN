#include "paddle/extension.h"
#include <vector>
#include <iostream>
#include <glog/logging.h>
#define CHECK_CUDA(x) PD_CHECK(x.is_gpu(),#x" must be a GPU Tensor.")
#define CHECK_INPUT(x) CHECK_CUDA(x)
#define DIVUP(m,n) ((m)/(n)+((m)%(n)>0))
 int THREADS_PER_BLOCK_NMS = sizeof(uint32_t ) * 8;

void boxesoverlapLauncher( int num_a,  float *boxes_a,  int num_b,  float *boxes_b, float *ans_overlap);
void boxesioubevLauncher( int num_a,  float *boxes_a,  int num_b,  float *boxes_b, float *ans_iou);
void nmsLauncher( float *boxes, uint32_t * mask, int boxes_num, float nms_overlap_thresh);
void nmsNormalLauncher( float *boxes, uint32_t * mask, int boxes_num, float nms_overlap_thresh);



std::vector<paddle::Tensor>  boxes_overlap_bev_gpu(const paddle::Tensor& boxes_a, 
const paddle::Tensor& boxes_b, const paddle::Tensor& ans_overlap){
    // params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
    // params boxes_b: (M, 5) 
    // params ans_overlap: (N, M)
    
    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_overlap);

    int num_a = boxes_a.shape()[0];
    int num_b = boxes_b.shape()[0];

    float * boxes_a_data = const_cast<float*>(boxes_a.data<float>());
    float * boxes_b_data = const_cast<float*>(boxes_b.data<float>());
    float * ans_overlap_data = const_cast<float*>(ans_overlap.data<float>());

    boxesoverlapLauncher(num_a, boxes_a_data, 
    num_b, boxes_b_data, ans_overlap_data);

    return {ans_overlap};
}


std::vector<std::vector<int64_t>>  boxes_overlap_bev_gpu_infershape(std::vector<int64_t> boxes_a_shape,
                                                                                                    std::vector<int64_t> boxes_b_shape,
                                                                                                    std::vector<int64_t> ans_overlap_shape){
                                                                                                        return {{boxes_b_shape[0],boxes_a_shape[0]}};
                                                                                                    }

std::vector<paddle::DataType>  boxes_overlap_bev_gpu_inferdtype(paddle::DataType boxes_a_dtype,
                                                                                                    paddle::DataType boxes_b_dtype,
                                                                                                    paddle::DataType ans_overlap_dtype){
                                                                                                        return {boxes_a_dtype};
                                                                                                    }
PD_BUILD_OP(boxes_overlap_bev_gpu)
    .Inputs({"BOXES_A", "BOXES_B","ANS_OVERLAP"})
    .Outputs({"Output"})
    .SetKernelFn(PD_KERNEL(boxes_overlap_bev_gpu))
    .SetInferShapeFn(PD_INFER_SHAPE(boxes_overlap_bev_gpu_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(boxes_overlap_bev_gpu_inferdtype));

std::vector<paddle::Tensor> boxes_iou_bev_gpu(const paddle::Tensor& boxes_a, const paddle::Tensor& boxes_b, const paddle::Tensor& ans_iou){
    // params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
    // params boxes_b: (M, 5) 
    // params ans_overlap: (N, M)
    
    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_iou);

    int num_a = boxes_a.shape()[0];
    int num_b = boxes_b.shape()[0];

    float * boxes_a_data = const_cast<float*>(boxes_a.data<float>());
    float * boxes_b_data = const_cast<float*>(boxes_b.data<float>());
    float * ans_iou_data = const_cast<float*>(ans_iou.data<float>());

    boxesioubevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

     return {ans_iou};
}

std::vector<std::vector<int64_t>>  boxes_iou_bev_gpu_infershape(std::vector<int64_t> boxes_a_shape,
                                                                                                    std::vector<int64_t> boxes_b_shape,
                                                                                                    std::vector<int64_t> ans_iou_shape){
                                                                                                        return {{boxes_a_shape[0],boxes_b_shape[0]}};
                                                                                                    }

std::vector<paddle::DataType>  boxes_iou_bev_gpu_inferdtype(paddle::DataType boxes_a_dtype,
                                                                                                    paddle::DataType boxes_b_dtype,
                                                                                                    paddle::DataType ans_overlap_dtype){
                                                                                                        return {boxes_a_dtype};
                                                                                                    }
PD_BUILD_OP(boxes_iou_bev_gpu)
    .Inputs({"BOXES_A", "BOXES_B","ANS_IOU"})
    .Outputs({"Output"})
    .SetKernelFn(PD_KERNEL(boxes_iou_bev_gpu))
    .SetInferShapeFn(PD_INFER_SHAPE(boxes_iou_bev_gpu_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(boxes_iou_bev_gpu_inferdtype));

std::vector<paddle::Tensor>  nms_gpu(const paddle::Tensor& boxes, const paddle::Tensor& keep, 
float nms_overlap_thresh){
    // params boxes: (N, 5) [x1, y1, x2, y2, ry]
    // params keep: (N)

    CHECK_INPUT(boxes);
    int boxes_num = boxes.shape()[0];
    float * boxes_data = const_cast<float*>(boxes.data<float>());
    int32_t * keep_data = const_cast<int32_t *>(keep.data<int32_t>());
    
    // std::cout<<nms_overlap_thresh_data<<std::endl;
    int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    uint32_t  *mask_data;
    cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(uint32_t));
    
    nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);
    

    std::vector<uint32_t > mask_cpu(boxes_num * col_blocks);
    cudaMemcpy(&mask_cpu[0],mask_data,boxes_num * col_blocks * sizeof(uint32_t),cudaMemcpyDeviceToHost);
    cudaFree(mask_data);
    
   
    uint32_t   remv_cpu[col_blocks];
    memset(remv_cpu, 0, col_blocks * sizeof(uint32_t));
    int num_to_keep = 0;
    std::vector<int32_t>  keep_data_cpu(boxes_num,0);
    cudaMemcpy(&keep_data_cpu[0],keep_data,boxes_num*sizeof(int32_t),cudaMemcpyDeviceToHost);
    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;
        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data_cpu[num_to_keep++] = i;
            uint32_t   *p =&mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }

    auto num_to_keep_tensor= paddle::full({1}, num_to_keep,paddle::DataType::INT32, paddle::GPUPlace());

    return {num_to_keep_tensor};
}

std::vector<std::vector<int64_t>>  nms_gpu_infershape(std::vector<int64_t>  boxes_shape,
                                                                                                    std::vector<int64_t>  keep_shape,
                                                                                                   float  nms_overlap_thresh){
                                                                                                        return {{1}};
                                                                                                    }

std::vector<paddle::DataType>  nms_gpu_inferdtype(paddle::DataType boxes_dtype,
                                                                                                    paddle::DataType keep_dtype
                                                                                                ){
                                                                                                        return {keep_dtype};
                                                                                                    }
PD_BUILD_OP(nms_gpu)
    .Inputs({"BOXES", "KEEP"})
    .Outputs({"Output"})
    .Attrs({"NMS_OVERLAP:float"})
    .SetKernelFn(PD_KERNEL(nms_gpu))
    .SetInferShapeFn(PD_INFER_SHAPE(nms_gpu_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(nms_gpu_inferdtype));

std::vector<paddle::Tensor> nms_normal_gpu(const paddle::Tensor& boxes, const paddle::Tensor& keep, 
float nms_overlap_thresh){
    // params boxes: (N, 5) [x1, y1, x2, y2, ry]
    // params keep: (N)

    CHECK_INPUT(boxes);

    int boxes_num = boxes.shape()[0];
    float * boxes_data = const_cast<float*>(boxes.data<float>());
    int32_t * keep_data = const_cast<int32_t*>(keep.data<int32_t>());
   
    int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    uint32_t  *mask_data = NULL;
    cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(uint32_t));
    nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);
    
    std::vector<int32_t  > mask_cpu(boxes_num * col_blocks);
    cudaMemcpy(&mask_cpu[0],mask_data,boxes_num * col_blocks * sizeof(uint32_t),cudaMemcpyDeviceToHost);
    cudaFree(mask_data);

    uint32_t    remv_cpu[col_blocks];
    memset(remv_cpu, 0, col_blocks * sizeof(uint32_t   ));

    int num_to_keep = 0;
    std::vector<int32_t>  keep_data_cpu(boxes_num,0);
    cudaMemcpy(&keep_data_cpu[0],keep_data,boxes_num*sizeof(int32_t),cudaMemcpyDeviceToHost);
    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data_cpu[num_to_keep++] = i;
            int32_t    *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }
    auto num_to_keep_tensor= paddle::full({1}, num_to_keep,paddle::DataType::INT32, paddle::GPUPlace());
    return {num_to_keep_tensor};
}

std::vector<std::vector<int64_t>>  nms_normal_gpu_infershape(std::vector<int64_t> boxes_shape,
                                                                                                    std::vector<int64_t> keep_shape,
                                                                                                    float  nms_overlap_thresh){
                                                                                                        return {{1}};
                                                                                                    }

std::vector<paddle::DataType>  nms_normal_gpu_inferdtype(paddle::DataType boxes_dtype,
                                                                                                    paddle::DataType keep_dtype){
                                                                                                        return {keep_dtype};
                                                                                                    }
PD_BUILD_OP(nms_normal_gpu)
    .Inputs({"BOXES", "KEEP"})
    .Outputs({"Output"})
    .Attrs({"NMS_OVERLAP:float"})
    .SetKernelFn(PD_KERNEL(nms_normal_gpu))
    .SetInferShapeFn(PD_INFER_SHAPE(nms_normal_gpu_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(nms_normal_gpu_inferdtype));