import paddle
import pointnet2

    # // points: (B, C, N)
    # // idx: (B, npoints)
    # // output:
    # //      out: (B, C, npoints)
# paddle.device.set_device('gpu')
# points=paddle.rand([4,128,16384],dtype='float32')
# idx=paddle.ones([4,1024,32],dtype='int32')
# out=paddle.rand([4,128,1024,32],dtype='float32')
# out=pointnet2.group_points_wrapper(points,idx,out,4,128,16384,1024,32)
# unknown=paddle.rand([4,16384,3])
# known=paddle.rand([4,1638,3])
# dist2=paddle.rand([4,16384,3])
# idx=paddle.ones([4,16384,3],dtype='int32')
# dist2,idx=pointnet2.three_nn_wrapper(unknown,known,dist2,idx,4,16384,1638)
# print(dist2.shape)
# print(idx.shape)

# points=paddle.rand([4,128,1024])
# idx=paddle.ones([4,16384,3],dtype='int32')
# weight=paddle.rand([4,16384,3])
# out=paddle.rand([4,128,16384])
# out=pointnet2.three_interpolate_wrapper(points,idx,weight,out,4,128,1024,16384)
# print(out.shape)

grad_out=paddle.rand([4,128,16384])
idx=paddle.ones([4,16384,3],dtype='int32')
weight=paddle.rand([4,16384,3])
grad_points=paddle.rand([4,128,1024])

grad_points=pointnet2.three_interpolate_grad_wrapper(grad_out,idx,weight,grad_points,4,128,16384,1024)
print(grad_points)

