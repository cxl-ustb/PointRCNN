#include "paddle/extension.h"
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "interpolate_gpu.h"
#include <glog/logging.h>

void init_glog(){
    if(!google::kLogSiteUninitialized)
        google::InitGoogleLogging("glog test main.cpp");
    google::SetLogFilenameExtension("log_");
    //会输出导致程序结束的信号,和google::InstallFailureWriter(&FatalMessageDump); 配合使用，可以在程序出现严重错误时将详细的错误信息打印出来
    google::InstallFailureSignalHandler(); //注册一下即可。默认是打印到stderr中，可以通过InstallFailureWriter更改输出目标。
    
    
    google::SetLogDestination(google::INFO, "log/"); // 把日志同时记录文件，最低级别为INFO，此时全部输出

    // 通过GFLAGS来设置参数，更多选项可以在logging.cc里面查询
    // 日志等级分为INFO, WARNING, ERROR, FATAL,如果是FATAL级别这直接运行报错
    FLAGS_stderrthreshold = google::INFO;    //INFO, WARNING, ERROR都输出，若为google::WARNING，则只输出WARNING, ERROR
    //google::SetStderrLogging(google::GLOG_INFO);
    FLAGS_colorlogtostderr = true;  //log为彩色
    
}


void close_glog(){
      google::ShutdownGoogleLogging();  //关闭log服务
}

std::vector<paddle::Tensor> three_nn_wrapper_fast(
   const paddle::Tensor& unknown_tensor, 
   const paddle::Tensor& known_tensor,  const paddle::Tensor& dist2_tensor,  
   const paddle::Tensor& idx_tensor,int b, int n, int m)
 {  init_glog();
    float *unknown =const_cast<float*>(unknown_tensor.data<float>());
    float *known = const_cast<float*>(known_tensor.data<float>());
    float *dist2 = const_cast<float*>(dist2_tensor.data<float>());
    int *idx = const_cast<int*>(idx_tensor.data<int>());
    three_nn_kernel_launcher_fast(b, n, m, unknown, known, dist2, idx);

    return {dist2_tensor,idx_tensor};
}



std::vector<paddle::Tensor> three_interpolate_wrapper_fast(
    const paddle::Tensor& points_tensor, 
     const paddle::Tensor& idx_tensor,  const paddle::Tensor& weight_tensor, 
      const paddle::Tensor& out_tensor,
     int b, int c, int m, int n)
     {
    float *points =const_cast<float*>(points_tensor.data<float>());
    float *weight = const_cast<float*>(weight_tensor.data<float>());
    float *out = const_cast<float*>(out_tensor.data<float>());
    int *idx = const_cast<int*>(idx_tensor.data<int>());

    three_interpolate_kernel_launcher_fast(b, c, m, n, points, idx, weight, out);
    return {out_tensor};
     }



std::vector<paddle::Tensor> three_interpolate_grad_wrapper_fast(
    const paddle::Tensor& grad_out_tensor, 
    const  paddle::Tensor& idx_tensor,  const paddle::Tensor& weight_tensor,  const paddle::Tensor& grad_points_tensor,int b, int c, int n, int m){
    float *grad_out = const_cast<float*>(grad_out_tensor.data<float>());
    float *weight = const_cast<float*>(weight_tensor.data<float>());
    float *grad_points = const_cast<float*>(grad_points_tensor.data<float>());
    int *idx = const_cast<int*>(idx_tensor.data<int>());

    three_interpolate_grad_kernel_launcher_fast(b, c, n, m, grad_out, idx, weight, grad_points);
    
    return {grad_points_tensor};
    }