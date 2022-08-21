import sys

import paddle
import paddle.nn as nn

sys.path.append('/home/cxl/PaperContest/PointRCNN_paddle')
sys.path.append('/home/cxl/PaperContest/PointRCNN_paddle/lib/utils/roipool3d')
from lib.config import cfg
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import numpy as np

class ProposalTargetLayer(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input_dict):
        roi_boxes3d, gt_boxes3d = input_dict['roi_boxes3d'], input_dict['gt_boxes3d']

        batch_rois, batch_gt_of_rois, batch_roi_iou = self.sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d)

        rpn_xyz, rpn_features = input_dict['rpn_xyz'], input_dict['rpn_features']
        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [
                                    paddle.unsqueeze(input_dict['rpn_intensity'],axis=2),
                                    paddle.unsqueeze(input_dict['seg_mask'],axis=2)
                                    ]
        else:
            pts_extra_input_list = [paddle.unsqueeze(input_dict['seg_mask'],axis=2)]


        if cfg.RCNN.USE_DEPTH:
            pts_depth = input_dict['pts_depth'] / 70.0 - 0.5
            pts_extra_input_list.append(paddle.unsqueeze(pts_depth,axis=2))


        pts_extra_input = paddle.concat(pts_extra_input_list,axis=2)

        # point cloud pooling
        pts_feature = paddle.concat((pts_extra_input,rpn_features),axis=2)

        pooled_features, pooled_empty_flag = \
            roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                          sampled_pt_num=cfg.RCNN.NUM_POINTS)

        sampled_pts, sampled_features = pooled_features[:, :, :, 0:3], pooled_features[:, :, :, 3:]

        # data augmentation
        if cfg.AUG_DATA:
            # data augmentation
            sampled_pts, batch_rois, batch_gt_of_rois = \
                self.data_augmentation(sampled_pts, batch_rois, batch_gt_of_rois)

        # canonical transformation
        batch_size = batch_rois.shape[0]
        roi_ry = batch_rois[:, :, 6] % (2 * np.pi)
        roi_center = batch_rois[:, :, 0:3]
        sampled_pts = sampled_pts -paddle.unsqueeze(roi_center,axis=2)# (B, M, 512, 3)
        batch_gt_of_rois[:, :, 0:3] = batch_gt_of_rois[:, :, 0:3] - roi_center
        batch_gt_of_rois[:, :, 6] = batch_gt_of_rois[:, :, 6] - roi_ry

        for k in range(batch_size):
            sampled_pts[k] = kitti_utils.rotate_pc_along_y_paddle(sampled_pts[k], batch_rois[k, :, 6])
            batch_gt_of_rois[k] = paddle.squeeze(kitti_utils.rotate_pc_along_y_paddle(paddle.unsqueeze(batch_gt_of_rois[k],axis=1),
                                                                      roi_ry[k]),axis=1)

        # regression valid mask
        valid_mask = (pooled_empty_flag == 0)
        reg_valid_mask = ((batch_roi_iou > cfg.RCNN.REG_FG_THRESH) & valid_mask).astype('long')

        # classification label
        batch_cls_label = (batch_roi_iou > cfg.RCNN.CLS_FG_THRESH).astype('long')
        invalid_mask = (batch_roi_iou > cfg.RCNN.CLS_BG_THRESH) & (batch_roi_iou < cfg.RCNN.CLS_FG_THRESH)
        batch_cls_label[valid_mask == 0] = -1
        batch_cls_label[invalid_mask > 0] = -1

        output_dict = {'sampled_pts': paddle.reshape(sampled_pts,[-1, cfg.RCNN.NUM_POINTS, 3]),
                       'pts_feature': sampled_features.reshape([-1, cfg.RCNN.NUM_POINTS, sampled_features.shape[3]]),
                       'cls_label': paddle.reshape(batch_cls_label,[-1]),
                       'reg_valid_mask':paddle.reshape(reg_valid_mask,[-1]),
                       'gt_of_rois':paddle.reshape(batch_gt_of_rois,[-1,7]),
                       'gt_iou':paddle.reshape(batch_roi_iou,[-1]),
                       'roi_boxes3d':paddle.reshape(batch_rois,[-1,7])}

        return output_dict

    def is_greater_thres(self,max_overlaps,g_thresh):
        max_overlaps_flag=paddle.zeros(max_overlaps.shape,dtype='bool')
        for i in range(len(max_overlaps)):
            if max_overlaps[i]>=g_thresh:
                max_overlaps_flag[i]=True
            else:
                max_overlaps_flag[i]=False
        return max_overlaps_flag


    def sample_rois_for_rcnn(self, roi_boxes3d, gt_boxes3d):
        """
        :param roi_boxes3d: (B, M, 7)
        :param gt_boxes3d: (B, N, 8) [x, y, z, h, w, l, ry, cls]
        :return
            batch_rois: (B, N, 7)
            batch_gt_of_rois: (B, N, 8)
            batch_roi_iou: (B, N)
        """
        batch_size = roi_boxes3d.shape[0]
        fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))

        batch_rois=paddle.zeros([batch_size, cfg.RCNN.ROI_PER_IMAGE, 7],dtype=gt_boxes3d.dtype)
        batch_gt_of_rois = paddle.zeros([batch_size, cfg.RCNN.ROI_PER_IMAGE, 7], dtype=gt_boxes3d.dtype)
        batch_roi_iou=paddle.zeros([batch_size, cfg.RCNN.ROI_PER_IMAGE],dtype=gt_boxes3d.dtype)

        for idx in range(batch_size):
            cur_roi, cur_gt = roi_boxes3d[idx], gt_boxes3d[idx]
            k = cur_gt.shape[0] - 1
            while cur_gt[k].sum() == 0:
                k -= 1


            cur_gt = cur_gt[:k + 1]

            # include gt boxes in the candidate rois

            iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)

            max_overlaps=paddle.max(iou3d,axis=1)
            gt_assignment=paddle.argmax(iou3d,axis=1)


            # sample fg, easy_bg, hard_bg
            fg_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)

            fg_inds=paddle.nonzero(max_overlaps >= fg_thresh).squeeze(-1)


            # TODO: this will mix the fg and bg when CLS_BG_THRESH_LO < iou < CLS_BG_THRESH
            easy_bg_inds =paddle.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH_LO)).squeeze(-1)
            hard_bg_inds =paddle.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH) &
                                         (max_overlaps >= cfg.RCNN.CLS_BG_THRESH_LO)).squeeze(-1)


            fg_num_rois = fg_inds.numel()
            bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                rand_num = paddle.to_tensor(np.random.permutation(fg_num_rois)).astype(gt_boxes3d.dtype)
                fg_inds = fg_inds[:int((rand_num[:fg_rois_per_this_image.numpy()[0]]).numpy()[0])]

                # sampling bg
                bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE - fg_rois_per_this_image
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                rand_num = np.floor(np.random.rand(cfg.RCNN.ROI_PER_IMAGE) * fg_num_rois.numpy[0])
                rand_num = paddle.to_tensor(rand_num).astype(gt_boxes3d.dtype)
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

                fg_rois_per_this_image = 0
            else:
                import pdb
                pdb.set_trace()
                raise NotImplementedError

            # augment the rois by noise
            roi_list, roi_iou_list, roi_gt_list = [], [], []
            if fg_rois_per_this_image > 0:
                fg_rois_src = cur_roi[fg_inds]
                gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]
                iou3d_src = max_overlaps[fg_inds]
                fg_rois, fg_iou3d = self.aug_roi_by_noise_paddle(fg_rois_src, gt_of_fg_rois, iou3d_src,
                                                                aug_times=cfg.RCNN.ROI_FG_AUG_TIMES)
                roi_list.append(fg_rois)
                roi_iou_list.append(fg_iou3d)
                roi_gt_list.append(gt_of_fg_rois)

            if bg_rois_per_this_image > 0:
                bg_rois_src = cur_roi[bg_inds]
                gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
                iou3d_src = max_overlaps[bg_inds]
                aug_times = 1 if cfg.RCNN.ROI_FG_AUG_TIMES > 0 else 0
                bg_rois, bg_iou3d = self.aug_roi_by_noise_paddle(bg_rois_src, gt_of_bg_rois, iou3d_src,
                                                                aug_times=aug_times)
                roi_list.append(bg_rois)
                roi_iou_list.append(bg_iou3d)
                roi_gt_list.append(gt_of_bg_rois)

            rois = paddle.concat(roi_list, 0)
            iou_of_rois = paddle.concat(roi_iou_list,0)
            gt_of_rois = paddle.concat(roi_gt_list, 0)

            batch_rois[idx] = rois
            batch_gt_of_rois[idx] = gt_of_rois
            batch_roi_iou[idx] = iou_of_rois

        return batch_rois, batch_gt_of_rois, batch_roi_iou

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image * cfg.RCNN.HARD_BG_RATIO)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = paddle.randint(low=0, high=hard_bg_inds.numel(), shape=(hard_bg_rois_num,)).astype('long')
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = paddle.randint(low=0, high=easy_bg_inds.numel(), shape=(easy_bg_rois_num,)).astype('long')
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = paddle.concat([hard_bg_inds, easy_bg_inds], axis=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = paddle.randint(low=0, high=hard_bg_inds.numel(), shape=(hard_bg_rois_num,)).astype('long')
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = paddle.randint(low=0, high=easy_bg_inds.numel(), shape=(easy_bg_rois_num,)).astype('long')
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    def aug_roi_by_noise_paddle(self, roi_boxes3d, gt_boxes3d, iou3d_src, aug_times=10):
        iou_of_rois = paddle.zeros([roi_boxes3d.shape[0]]).astype(gt_boxes3d.dtype)
        pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)

        for k in range(roi_boxes3d.shape[0]):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]
            try:
                gt_box3d = paddle.reshape(gt_boxes3d[k],[1,7])
            except:
                print(gt_boxes3d[k].shape)
            aug_box3d = roi_box3d
            keep = True
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                    keep = True
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                    keep = False
                aug_box3d = paddle.reshape(aug_box3d,[1,7])
                iou3d = iou3d_utils.boxes_iou3d_gpu(aug_box3d, gt_box3d)
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = paddle.reshape(aug_box3d,[-1])
            if cnt == 0 or keep:
                iou_of_rois[k] = iou3d_src[k]
            else:
                iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    @staticmethod
    def random_aug_box3d(box3d):
        """
        :param box3d: (7) [x, y, z, h, w, l, ry]
        random shift, scale, orientation
        """
        paddle.device.set_device('gpu')
        if cfg.RCNN.REG_AUG_METHOD == 'single':
            pos_shift = (paddle.rand([3]) - 0.5)  # [-0.5 ~ 0.5]
            hwl_scale = (paddle.rand([3]) - 0.5) / (0.5 / 0.15) + 1.0  #
            angle_rot = (paddle.rand([1]) - 0.5) / (0.5 / (np.pi / 12))  # [-pi/12 ~ pi/12]
            aug_box3d = paddle.concat([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot], 0)
            return aug_box3d
        elif cfg.RCNN.REG_AUG_METHOD == 'multiple':
            # pos_range, hwl_range, angle_range, mean_iou
            range_config = [[0.2, 0.1, np.pi / 12, 0.7],
                            [0.3, 0.15, np.pi / 12, 0.6],
                            [0.5, 0.15, np.pi / 9, 0.5],
                            [0.8, 0.15, np.pi / 6, 0.3],
                            [1.0, 0.15, np.pi / 3, 0.2]]


            idx=paddle.randint(0,len(range_config),[1])[0].astype('long')
            pos_shift = ((paddle.rand([3]) - 0.5) / 0.5) * range_config[idx][0]
            hwl_scale = ((paddle.rand([3]) - 0.5) / 0.5) * range_config[idx][1] + 1.0
            angle_rot = ((paddle.rand([1]) - 0.5) / 0.5) * range_config[idx][2]

            aug_box3d = paddle.concat([box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot], 0)
            return aug_box3d

        elif cfg.RCNN.REG_AUG_METHOD == 'normal':
            x_shift = np.random.normal(loc=0, scale=0.3)
            y_shift = np.random.normal(loc=0, scale=0.2)
            z_shift = np.random.normal(loc=0, scale=0.3)
            h_shift = np.random.normal(loc=0, scale=0.25)
            w_shift = np.random.normal(loc=0, scale=0.15)
            l_shift = np.random.normal(loc=0, scale=0.5)
            ry_shift = ((paddle.rand([1]) - 0.5) / 0.5) * np.pi / 12

            aug_box3d = np.array([box3d[0] + x_shift, box3d[1] + y_shift, box3d[2] + z_shift, box3d[3] + h_shift,
                                  box3d[4] + w_shift, box3d[5] + l_shift, box3d[6] + ry_shift], dtype=np.float32)
            aug_box3d = paddle.to_tensor(aug_box3d).astype(box3d)
            return aug_box3d
        else:
            raise NotImplementedError

    def data_augmentation(self, pts, rois, gt_of_rois):
        """
        :param pts: (B, M, 512, 3)
        :param rois: (B, M. 7)
        :param gt_of_rois: (B, M, 7)
        :return:
        """
        paddle.device.set_device('gpu')
        batch_size, boxes_num = pts.shape[0], pts.shape[1]

        # rotation augmentation
        angles = (paddle.rand([batch_size, boxes_num]) - 0.5 / 0.5) * (np.pi / cfg.AUG_ROT_RANGE)

        # calculate gt alpha from gt_of_rois
        temp_x, temp_z, temp_ry = gt_of_rois[:, :, 0], gt_of_rois[:, :, 2], gt_of_rois[:, :, 6]
        temp_beta = paddle.atan2(temp_z, temp_x)
        gt_alpha = -paddle.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry  # (B, M)

        temp_x, temp_z, temp_ry = rois[:, :, 0], rois[:, :, 2], rois[:, :, 6]
        temp_beta = paddle.atan2(temp_z, temp_x)
        roi_alpha = -paddle.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry  # (B, M)

        for k in range(batch_size):
            pts[k] = kitti_utils.rotate_pc_along_y_paddle(pts[k], angles[k])
            gt_of_rois[k] =paddle.squeeze(kitti_utils.rotate_pc_along_y_paddle(paddle.unsqueeze(gt_of_rois[k],1), angles[k]),axis=1)
            rois[k] =paddle.squeeze(kitti_utils.rotate_pc_along_y_paddle(paddle.unsqueeze(rois[k],1), angles[k]),1)

            # calculate the ry after rotation
            temp_x, temp_z = gt_of_rois[:, :, 0], gt_of_rois[:, :, 2]
            temp_beta = paddle.atan2(temp_z, temp_x)
            gt_of_rois[:, :, 6] = paddle.sign(temp_beta) * np.pi / 2 + gt_alpha - temp_beta

            temp_x, temp_z = rois[:, :, 0], rois[:, :, 2]
            temp_beta = paddle.atan2(temp_z, temp_x)
            rois[:, :, 6] = paddle.sign(temp_beta) * np.pi / 2 + roi_alpha - temp_beta

        # scaling augmentation
        scales= (1 + ((paddle.rand((batch_size, boxes_num)) - 0.5) / 0.5) * 0.05)
        pts=pts*scales.unsqueeze(2).unsqueeze(3)

        gt_of_rois=gt_of_rois.numpy()
        scales=scales.numpy()
        # gt_of_rois[:, :, 0:6] = paddle.unsqueeze(gt_of_rois[:, :, 0:6] * scales,2)
        gt_of_rois[:, :, 0:6]=gt_of_rois[:, :, 0:6]*scales[:,:,np.newaxis]
        gt_of_rois=paddle.to_tensor(gt_of_rois)

        rois=rois.numpy()
        rois[:, :, 0:6]=rois[:, :, 0:6]*scales[:,:,np.newaxis]
        rois = paddle.to_tensor(rois)

        # flip augmentation
        flip_flag = paddle.sign(paddle.rand((batch_size, boxes_num)) - 0.5)
        pts[:, :, :, 0] = pts[:, :, :, 0] * flip_flag.unsqueeze(2)
        gt_of_rois[:, :, 0] = gt_of_rois[:, :, 0] * flip_flag
        # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
        src_ry = gt_of_rois[:, :, 6]
        ry = (flip_flag == 1).astype('float32') * src_ry + (flip_flag == -1).astype('float32') * (paddle.sign(src_ry) * np.pi - src_ry)
        gt_of_rois[:, :, 6] = ry

        rois[:, :, 0] = rois[:, :, 0] * flip_flag
        # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
        src_ry = rois[:, :, 6]
        ry = (flip_flag == 1).astype('float32') * src_ry + (flip_flag == -1).astype('float32') * (paddle.sign(src_ry) * np.pi - src_ry)
        rois[:, :, 6] = ry

        return pts, rois, gt_of_rois
