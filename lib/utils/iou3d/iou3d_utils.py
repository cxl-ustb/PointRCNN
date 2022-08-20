import paddle
import iou3d
import sys
sys.path.append('/home/cxl/PaperContest/PointRCNN_paddle')
import lib.utils.kitti_utils as kitti_utils
import numpy as np


def boxes_iou_bev(boxes_a,boxes_b):
    '''
    :param boxes_a: (M,5)
    :param boxes_b: (N,5)
    :return:
           ans_iou:(M,N)
    '''
    paddle.device.set_device('gpu')
    ans_iou=paddle.zeros([boxes_a.shape[0],boxes_b.shape[0]],dtype='float32')
    ans_iou=iou3d.boxes_iou_bev_gpu(boxes_a,boxes_b,ans_iou)
    return ans_iou

def boxes_iou3d_gpu(boxes_a,boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """

    paddle.device.set_device('gpu')
    boxes_a_bev=kitti_utils.boxes3d_to_bev_paddle(boxes_a)
    boxes_b_bev=kitti_utils.boxes3d_to_bev_paddle(boxes_b)

    #bev overlap
    overlaps_bev=paddle.zeros([boxes_b.shape[0],boxes_a.shape[0]],dtype='float32')
    overlaps_bev=iou3d.boxes_overlap_bev_gpu(boxes_a_bev,boxes_b_bev,overlaps_bev)

    #height overlap
    boxes_a_height_min=paddle.reshape((boxes_a[:,1]-boxes_a[:,3]),[-1,1])
    boxes_a_height_max=paddle.reshape((boxes_a[:,1]),[-1,1])
    boxes_b_height_min=paddle.reshape((boxes_b[:,1]-boxes_b[:,3]),[-1,1])
    boxes_b_height_max=paddle.reshape((boxes_b[:,1]),[-1,1])

    max_of_min=paddle.max(paddle.concat((boxes_a_height_min,boxes_b_height_min)))
    min_of_max=paddle.min(paddle.concat((boxes_a_height_max,boxes_b_height_max)))
    overlaps_h=paddle.clip(min_of_max-max_of_min,min=0)

    #3d iou
    # np.expand_dims(vol_a, -1).repeat(, axis=0)

    overlaps_3d=overlaps_bev*overlaps_h
    vol_a=paddle.reshape((boxes_a[:,3]*boxes_a[:,4]*boxes_a[:,5]),[-1,1]).numpy()
    vol_b=paddle.reshape((boxes_b[:,3]*boxes_b[:,4]*boxes_b[:,5]),[-1,1]).numpy()
    vol_a=paddle.to_tensor(vol_a.repeat(overlaps_3d.shape[0],-1).T)
    vol_b=paddle.to_tensor(vol_b.repeat(overlaps_3d.shape[1],-1))
    iou_3d=overlaps_3d/paddle.clip(vol_a+vol_b-overlaps_3d,min=1e-7)
    return iou_3d

def nms_gpu(boxes,scores,thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    paddle.device.set_device('gpu')
    order=paddle.argsort(scores,axis=0,descending=True)
    boxes=boxes[order]

    keep=paddle.zeros([boxes.shape[0]],dtype='int32')
    num_out=iou3d.nms_gpu(boxes,keep,0.3).numpy()
    return order[keep[:num_out[0]]]

def nms_normal_gpu(boxes,scores,thresh):
    """
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    paddle.device.set_device('gpu')
    order=paddle.argsort(scores,axis=0,descending=True)
    boxes=boxes[order]

    keep=paddle.zeros([boxes.shape[0]],dtype='int32')
    num_out=iou3d.nms_normal_gpu(boxes,keep,thresh).numpy()
    return order[keep[:num_out[0]]]

if __name__ == '__main__':
    """
        :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
        :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
        :return:
            ans_iou: (M, N)
        """
    paddle.device.set_device('gpu')
    boxes_a=paddle.randn([16389,7],dtype='float32')
    boxes_b=paddle.randn([1638,7],dtype='float32')
    ans=boxes_iou3d_gpu(boxes_a,boxes_b)
    print(ans)
    



