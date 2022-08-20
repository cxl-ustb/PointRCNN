import paddle
import roipool3d
import numpy as np
import sys
sys.path.append('/home/cxl/PaperContest/PointRCNN_paddle')
import lib.utils.kitti_utils as kitti_utils

def roipool3d_gpu(pts,pts_feature,boxes3d,pool_extra_width,sampled_pt_num=512):
    '''
    :param pts: (B.N,3)
    :param pts_feature: (B,N,C)
    :param boxes3d: (B.M,7)
    :param pool_extra_width: float
    :param sampled_pt_num: int
    :return:
        pooled_features:(B,M,512,3+C）
        pooled_empty_flag:(B,M)
    '''
    batch_size,boxes_num,feature_len=pts.shape[0],boxes3d.shape[1],pts_feature.shape[2]
    pooled_boxes3d=paddle.reshape(kitti_utils.enlarge_box3d(paddle.reshape(boxes3d,[-1,7]),pool_extra_width),[batch_size,-1,7])
    
    paddle.device.set_device("gpu")
    pooled_features=paddle.zeros([batch_size,boxes_num,sampled_pt_num,3+pts_feature.shape[2]],dtype='float32')
    pooled_empty_flag=paddle.zeros([batch_size,boxes_num],dtype='int32')

    pooled_features=roipool3d.forward(pts,pooled_boxes3d,pts_feature,pooled_features,pooled_empty_flag)

    return pooled_features,pooled_empty_flag



def pts_in_boxes3d_cpu(pts, boxes3d):
    """
    :param pts: (N, 3) in rect-camera coords
    :param boxes3d: (M, 7)
    :return: boxes_pts_mask_list: (M), list with [(N), (N), ..]
    """
    if not pts.place.is_gpu_place():
        pts_flag=paddle.zeros([boxes3d.shape[0],pts.shape[0]],dtype='int64')
        pts_flag=roipool3d.pts_in_boxes3d_cpu(pts_flag,pts,boxes3d)

        boxes_pts_mask_list=[]
        for k in range(0,boxes3d.shape[0]):
            cur_mask=pts_flag[k]>0
            boxes_pts_mask_list.append(cur_mask)
        return boxes_pts_mask_list

    else:
        raise NotImplementedError


def roipool_pc_cpu(pts,pts_feature,boxes3d,sampled_pt_num):
    '''
    :param pts: (N,3)
    :param pts_feature: (N,C)
    :param boxes3d: (M,7)
    :param sampled_pt_num: int
    :return:
    '''
    assert pts.shape[0] == pts_feature.shape[0] and pts.shape[1] == 3, '%s %s' % (pts.shape, pts_feature.shape)
    assert pts.place.is_gpu_place() is False
    pooled_pts=paddle.zeros([boxes3d.shape[0],sampled_pt_num,3],dtype='float32')
    pooled_features=paddle.zeros([boxes3d.shape[0],sampled_pt_num,pts_feature.shape[1]],dtype='float32')
    pooled_empty_flag=paddle.zeros([boxes3d.shape[0]],dtype='int32')
    pooled_features=roipool3d.roipool3d_cpu(pts,boxes3d,pts_feature,pooled_pts,pooled_features,pooled_empty_flag)
    return pooled_pts,pooled_features,pooled_empty_flag


def roipool3d_cpu(boxes3d, pts, pts_feature, pts_extra_input,
                  pool_extra_width, sampled_pt_num=512,
                  canonical_transform=True):
    """
    :param boxes3d: (N, 7)
    :param pts: (N, 3)
    :param pts_feature: (N, C)
    :param pts_extra_input: (N, C2)
    :param pool_extra_width: constant
    :param sampled_pt_num: constant
    :return:
    """
    pooled_boxes3d = kitti_utils.enlarge_box3d(boxes3d, pool_extra_width)

    pts_feature_all = np.concatenate((pts_extra_input, pts_feature), axis=1)

    #  Note: if pooled_empty_flag[i] > 0, the pooled_pts[i], pooled_features[i] will be zero
    pooled_pts, pooled_features, pooled_empty_flag = \
        roipool_pc_cpu(paddle.to_tensor(pts), paddle.to_tensor(pts_feature_all),
                       paddle.to_tensor(pooled_boxes3d), sampled_pt_num)

    extra_input_len = pts_extra_input.shape[1]
    sampled_pts_input=paddle.concat([pooled_pts, pooled_features[:, :, 0:extra_input_len]],axis=2)

    sampled_pts_feature = pooled_features[:, :, extra_input_len:].numpy()

    if canonical_transform:
        # Translate to the roi coordinates
        roi_ry = boxes3d[:, 6] % (2 * np.pi)  # 0~2pi
        roi_center = boxes3d[:, 0:3]

        # shift to center
        sampled_pts_input[:, :, 0:3] = sampled_pts_input[:, :, 0:3] - roi_center[:, np.newaxis, :]
        for k in range(sampled_pts_input.shape[0]):
            sampled_pts_input[k] = kitti_utils.rotate_pc_along_y(sampled_pts_input[k], roi_ry[k])

        return sampled_pts_input, sampled_pts_feature

    return sampled_pts_input, sampled_pts_feature, pooled_empty_flag.numpy()

if __name__ == '__main__':
    
# def roipool_pc_cpu(pts,pts_feature,boxes3d,sampled_pt_num):
    '''
    :param pts: (N,3)
    :param pts_feature: (N,C)
    :param boxes3d: (M,7)
    :param sampled_pt_num: int
    :return:
    '''
    paddle.device.set_device("cpu")
    pts=paddle.ones([16683,3])
    pts_feature=paddle.ones([16683,7])
    boxes3d=paddle.ones([1683,7])
    pooled_pts,pooled_features,pooled_empty_flag=roipool_pc_cpu(pts,pts_feature,boxes3d,512)
    print(pooled_pts,pooled_features)
    
    
