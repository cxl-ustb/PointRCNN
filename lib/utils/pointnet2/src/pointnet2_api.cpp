#include "paddle/extension.h"
#include "sampling_gpu.h"
#include "ball_query_gpu.h"
#include "group_points_gpu.h"
#include "interpolate_gpu.h"
std::vector<std::vector<int64_t>>  gather_points_wrapper_infershape(
                        std::vector<int64_t> points_tensor_shape,
                        std::vector<int64_t>  idx_tensor_shape,
                        std::vector<int64_t>  out_tensor_shape,
                        int b, int c, int n, int npoints){
    return {{points_tensor_shape[0],points_tensor_shape[1],idx_tensor_shape[1]}};
                                                                                                    }

std::vector<paddle::DataType>  gather_points_wrapper_inferdtype(
                        paddle::DataType points_tensor_dtype,
                        paddle::DataType idx_tensor_dtype,
                        paddle::DataType out_tensor_dtype
                                                                                                ){
                        return {points_tensor_dtype};
                                                                                                    }
PD_BUILD_OP(gather_points_wrapper)
    .Inputs({"POINTS", "IDX","OUT"})
    .Outputs({"Output"})
    .Attrs({"B:int","C:int","N:int","NPOINTS:int"})
    .SetKernelFn(PD_KERNEL(gather_points_wrapper_fast))
    .SetInferShapeFn(PD_INFER_SHAPE(gather_points_wrapper_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(gather_points_wrapper_inferdtype));


std::vector<std::vector<int64_t>>  gather_points_grad_wrapper_fast_infershape(
                                                                                                     
                                                                                                    std::vector<int64_t> grad_out_tensor_shape,
                                                                                                    std::vector<int64_t>  idx_tensor_shape,
                                                                                                   std::vector<int64_t>  grad_points_tensor_shape,
                                                                                                   int b, int c, int n, int npoints){
                                                                                                        return {{grad_out_tensor_shape[0],grad_out_tensor_shape[1],n}};
                                                                                                    }

std::vector<paddle::DataType>  gather_points_grad_wrapper_fast_inferdtype(
                                                                    paddle::DataType grad_out_tensor_dtype,
                                                                    paddle::DataType idx_tensor_dtype,
                                                                    paddle::DataType grad_points_tensor_dtype
                                                                                                ){
                                                                                                        return {grad_out_tensor_dtype};
                                                                                                    }
PD_BUILD_OP(gather_points_grad_wrapper)
    .Inputs({"GRAD_OUT", "IDX","GRAD_POINTS"})
    .Outputs({"Output"})
    .Attrs({"B:int","C:int","N:int","NPOINTS:int"})
    .SetKernelFn(PD_KERNEL(gather_points_grad_wrapper_fast))
    .SetInferShapeFn(PD_INFER_SHAPE(gather_points_grad_wrapper_fast_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(gather_points_grad_wrapper_fast_inferdtype));

std::vector<std::vector<int64_t>>  furthest_point_sampling_wrapper_infershape(
                                                                                                    std::vector<int64_t>  points_tensor_shape,
                                                                                                    std::vector<int64_t>  temp_tensor_shape,
                                                                                                   std::vector<int64_t>  idx_tensor_shape,
                                                                                                      int b, int n, int m){
                                                                                                        return {{points_tensor_shape[0],m}};
                                                                                                    }

std::vector<paddle::DataType>  furthest_point_sampling_wrapper_inferdtype(
                                                                    paddle::DataType  points_tensor_dtype,
                                                                    paddle::DataType temp_tensor_dtype,
                                                                    paddle::DataType idx_tensor_dtype   
                                                                                                ){
                                                                                                        return {points_tensor_dtype};
                                                                                                    }
PD_BUILD_OP(furthest_point_sampling_wrapper)
    .Inputs({"POINTS", "TEMP","IDX"})
    .Outputs({"Output"})
    .Attrs({"B:int","N:int","M:int"})
    .SetKernelFn(PD_KERNEL(furthest_point_sampling_wrapper))
    .SetInferShapeFn(PD_INFER_SHAPE(furthest_point_sampling_wrapper_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(furthest_point_sampling_wrapper_inferdtype));

std::vector<std::vector<int64_t>>  ball_query_wrapper_fast_infershape(
                                                                                                    std::vector<int64_t> new_xyz_tensor_shape,
                                                                                                    std::vector<int64_t>  xyz_tensor_shape,
                                                                                                   std::vector<int64_t>  idx_tensor_shape,
                                                                                                     int b, int n, int m, float radius, int nsample){
                                                                                                        return {{xyz_tensor_shape[0],m,nsample}};
                                                                                                    }

std::vector<paddle::DataType>  ball_query_wrapper_fast_inferdtype(
                                                                    paddle::DataType new_xyz_tensor_dtype,
                                                                    paddle::DataType xyz_tensor_dtype,
                                                                    paddle::DataType idx_tensor_dtype
                                                                                                ){
                                                                                                        return {xyz_tensor_dtype};
                                                                                                    }
PD_BUILD_OP(ball_query_wrapper)
    .Inputs({"NEW_XYZ", "XYZ","IDX"})
    .Outputs({"Output"})
    .Attrs({"B:int","N:int","M:int","RADIUS:float","NSAMPLE:int"})
    .SetKernelFn(PD_KERNEL(ball_query_wrapper_fast))
    .SetInferShapeFn(PD_INFER_SHAPE(ball_query_wrapper_fast_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ball_query_wrapper_fast_inferdtype));

    std::vector<std::vector<int64_t>>  group_points_wrapper_fast_infershape(
                        std::vector<int64_t> points_tensor_shape,
                        std::vector<int64_t>  idx_tensor_shape,
                        std::vector<int64_t>  out_tensor_shape,
                        int b, int c, int n, int npoints,int nsample){
    return {{points_tensor_shape[0],points_tensor_shape[1],npoints,nsample}};
                                                                                                    }

std::vector<paddle::DataType>  group_points_wrapper_fast_inferdtype(
                        paddle::DataType points_tensor_dtype,
                        paddle::DataType idx_tensor_dtype,
                        paddle::DataType out_tensor_dtype
                                                                                                ){
                        return {points_tensor_dtype};
                                                                                                    }
PD_BUILD_OP(group_points_wrapper)
    .Inputs({"POINTS", "IDX","OUT"})
    .Outputs({"Output"})
    .Attrs({"B:int","C:int","N:int","NPOINTS:int","NSAMPLE:int"})
    .SetKernelFn(PD_KERNEL(group_points_wrapper_fast))
    .SetInferShapeFn(PD_INFER_SHAPE(group_points_wrapper_fast_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(group_points_wrapper_fast_inferdtype));

std::vector<std::vector<int64_t>>  group_points_grad_wrapper_fast_infershape(
                        std::vector<int64_t> grad_out_tensor_shape,
                        std::vector<int64_t>  idx_tensor_shape,
                        std::vector<int64_t>  grad_points_tensor_shape,
                         int b, int c, int n, int npoints, int nsample){
    return {{grad_out_tensor_shape[0],grad_out_tensor_shape[1],n}};
                                                                                                    }

std::vector<paddle::DataType>  group_points_grad_wrapper_fast_inferdtype(
                        paddle::DataType grad_out_tensor_dtype,
                        paddle::DataType idx_tensor_dtype,
                        paddle::DataType grad_points_tensor_dtype
                                                                                                ){
                        return {grad_out_tensor_dtype};
                                                                                                    }
PD_BUILD_OP(group_points_grad_wrapper)
    .Inputs({"GRAD_OUT", "IDX","GRAD_POINTS"})
    .Outputs({"Output"})
    .Attrs({"B:int","C:int","N:int","NPOINTS:int","NSAMPLE:int"})
    .SetKernelFn(PD_KERNEL(group_points_grad_wrapper_fast))
    .SetInferShapeFn(PD_INFER_SHAPE(group_points_grad_wrapper_fast_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(group_points_grad_wrapper_fast_inferdtype));

    std::vector<std::vector<int64_t>>  three_nn_wrapper_fast_infershape(
                        std::vector<int64_t> unknown_tensor_shape,
                        std::vector<int64_t>  known_tensor_shape,
                        std::vector<int64_t>  dist2_tensor_shape,std::vector<int64_t>  idx_tensor_shape,
                         int b, int n, int m){
    return {dist2_tensor_shape,idx_tensor_shape};
                                                                                                    }

std::vector<paddle::DataType>  three_nn_wrapper_fast_inferdtype(
                        paddle::DataType unknown_tensor_dtype,
                        paddle::DataType known_tensor_dtype,
                        paddle::DataType dist2_tensor_dtype,
                        paddle::DataType idx_tensor_dtype
                                                                                                ){
                        return {dist2_tensor_dtype,idx_tensor_dtype};
                                                                                                    }
PD_BUILD_OP(three_nn_wrapper)
    .Inputs({"UNKNOWN", "KNOWN","DIST2","IDX"})
    .Outputs({"Output1","OUTPUT2"})
    .Attrs({"B:int","N:int","M:int"})
    .SetKernelFn(PD_KERNEL(three_nn_wrapper_fast))
    .SetInferShapeFn(PD_INFER_SHAPE( three_nn_wrapper_fast_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(three_nn_wrapper_fast_inferdtype));


std::vector<std::vector<int64_t>>  three_interpolate_wrapper_infershape(
                        std::vector<int64_t> points_tensor_shape,
                        std::vector<int64_t>  idx_tensor_shape,
                        std::vector<int64_t>  weight_tensor_shape,std::vector<int64_t>  out_tensor_shape,
                          int b, int c, int m, int n){
    return {{points_tensor_shape[0],points_tensor_shape[1],idx_tensor_shape[1]}};
                                                                                                    }

std::vector<paddle::DataType> three_interpolate_wrapper_inferdtype(
                        paddle::DataType points_tensor_dtype,
                        paddle::DataType idx_tensor_dtype,
                        paddle::DataType weight_tensor_dtype,
                        paddle::DataType out_tensor_dtype
                                                                                                ){
                        return {out_tensor_dtype};
                                                                                                    }
PD_BUILD_OP( three_interpolate_wrapper)
    .Inputs({"POINTS", "IDX","WEIGHT","OUT"})
    .Outputs({"Output"})
    .Attrs({"B:int","C:int","M:int","N:int"})
    .SetKernelFn(PD_KERNEL( three_interpolate_wrapper_fast))
    .SetInferShapeFn(PD_INFER_SHAPE( three_interpolate_wrapper_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(three_interpolate_wrapper_inferdtype));



    std::vector<std::vector<int64_t>>  three_interpolate_grad_wrapper_infershape(
                        std::vector<int64_t> grad_out_tensor_shape,
                        std::vector<int64_t>  idx_tensor_shape,
                        std::vector<int64_t>  weight_tensor_shape,
                        std::vector<int64_t>  grad_points_tensor_shape,
                          int b, int c, int n, int m){
    return {{grad_out_tensor_shape[0],grad_out_tensor_shape[1],m}};
                                                                                                    }

std::vector<paddle::DataType> three_interpolate_grad_wrapper_inferdtype(
                        paddle::DataType grad_out_tensor_dtype,
                        paddle::DataType idx_tensor_dtype,
                        paddle::DataType weight_tensor_dtype,
                        paddle::DataType grad_points_tensor_dtype
                                                                                                ){
                        return {grad_points_tensor_dtype};
                                                                                                    }
PD_BUILD_OP(three_interpolate_grad_wrapper)
    .Inputs({"GRAD_OUT", "IDX","WEIGHT","GRAD_POINTS"})
    .Outputs({"Output"})
    .Attrs({"B:int","C:int","N:int","M:int"})
    .SetKernelFn(PD_KERNEL(three_interpolate_grad_wrapper_fast))
    .SetInferShapeFn(PD_INFER_SHAPE(three_interpolate_grad_wrapper_infershape))
    .SetInferDtypeFn(PD_INFER_DTYPE(three_interpolate_grad_wrapper_inferdtype));



