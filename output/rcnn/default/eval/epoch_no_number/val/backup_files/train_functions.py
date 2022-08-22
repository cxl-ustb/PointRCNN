import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
from collections import namedtuple


def model_joint_fn_decorator():
    paddle.device.set_device('gpu')
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', 'disp_dict'])
    MEAN_SIZE = paddle.to_tensor(cfg.CLS_MEAN_SIZE[0])

    def model_fn(model, data):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = paddle.to_tensor(rpn_cls_label).astype('long')
                rpn_reg_label = paddle.to_tensor(rpn_reg_label).astype('long')

            inputs = paddle.to_tensor(pts_input).astype('float32')
            gt_boxes3d = paddle.to_tensor(gt_boxes3d).astype('float32')
            input_data = {'pts_input': inputs, 'gt_boxes3d': gt_boxes3d}
        else:
            input_data = {}
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = paddle.to_tensor(val).astype('float32')
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = paddle.concat((input_data['pts_input'], input_data['pts_features']), -1)
                input_data['pts_input'] = pts_input

        ret_dict = model(input_data)

        tb_dict = {}
        disp_dict = {}
        loss = 0
        if cfg.RPN.ENABLED and not cfg.RPN.FIXED:
            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
            rpn_loss = get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict)
            loss += rpn_loss
            disp_dict['rpn_loss'] = rpn_loss.item(0)

        if cfg.RCNN.ENABLED:
            rcnn_loss = get_rcnn_loss(model, ret_dict, tb_dict)
            disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']
            loss += rcnn_loss

        disp_dict['loss'] = loss.item(0)

        return ModelReturn(loss, tb_dict, disp_dict)

    def get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict):
       
        rpn_cls_loss_func = model.rpn.rpn_cls_loss_func

        rpn_cls_label_flat = paddle.reshape(rpn_cls_label,[-1])
        rpn_cls_flat = paddle.reshape(rpn_cls,[-1])
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).astype('float32')
            pos = (rpn_cls_label_flat > 0).astype('float32')
            neg = (rpn_cls_label_flat == 0).astype('float32')
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / paddle.clip(pos_normalizer, min=1.0)
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item(0)
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item(0)

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = paddle.ones([rpn_cls_flat.shape[0]])
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).astype('float32')
            batch_loss_cls = F.binary_cross_entropy(paddle.nn.functional.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight=weight, reduction='none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).astype('float32')
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / paddle.clip(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.shape[0] * rpn_reg.shape[1]
        fg_sum = fg_mask.astype('long').sum().item(0)
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(paddle.reshape(rpn_reg,[point_num,-1])[fg_mask],
                                        paddle.reshape(rpn_reg_label,[point_num, 7])[fg_mask],
                                        loc_scope=cfg.RPN.LOC_SCOPE,
                                        loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size=MEAN_SIZE,
                                        get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                                        get_y_by_bin=False,
                                        get_ry_fine=False)

            loss_size = 3 * loss_size  # consistent with old codes
            rpn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]

        tb_dict.update({'rpn_loss_cls': rpn_loss_cls.item(0), 'rpn_loss_reg': rpn_loss_reg.item(0),
                        'rpn_loss': rpn_loss.item(0), 'rpn_fg_sum': fg_sum, 'rpn_loss_loc': loss_loc.item(0),
                        'rpn_loss_angle': loss_angle.item(0), 'rpn_loss_size': loss_size.item(0)})

        return rpn_loss

    def get_rcnn_loss(model, ret_dict, tb_dict):
        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']

        cls_label = ret_dict['cls_label'].astype('float32')
        reg_valid_mask = ret_dict['reg_valid_mask']
        roi_boxes3d = ret_dict['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = ret_dict['gt_of_rois']
        pts_input = ret_dict['pts_input']

      
        cls_loss_func = model.rcnn_net.cls_loss_func

        cls_label_flat = paddle.reshape(cls_label,[-1])

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            rcnn_cls_flat = paddle.reshape(rcnn_cls,[-1])

            cls_target = (cls_label_flat > 0).astype('float32')
            pos = (cls_label_flat > 0).astype('float32')
            neg = (cls_label_flat == 0).astype('float32')
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / paddle.clip(pos_normalizer, min=1.0)

            rcnn_loss_cls = cls_loss_func(rcnn_cls_flat, cls_target, cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rcnn_loss_cls_pos.item(0)
            tb_dict['rpn_loss_cls_neg'] = rcnn_loss_cls_neg.item(0)

        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = paddle.reshape(rcnn_cls,[-1])
            batch_loss_cls = F.binary_cross_entropy(paddle.nn.functional.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
            cls_valid_mask = (cls_label_flat >= 0).astype('float32')
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / paddle.clip(cls_valid_mask.sum(), min=1.0)

        elif cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            rcnn_cls_reshape = paddle.reshape(rcnn_cls,[rcnn_cls.shape[0],-1])
            cls_target = cls_label_flat.astype('long')
            cls_valid_mask = (cls_label_flat >= 0).astype('float32')

            batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target)
            normalizer = paddle.clip(cls_valid_mask.sum(), min=1.0)
            rcnn_loss_cls = (batch_loss_cls.mean(1) * cls_valid_mask).sum() / normalizer

        else:
            raise NotImplementedError

        # rcnn regression loss
        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.astype('long').sum().item(0)
        if fg_sum != 0:
            all_anchor_size = roi_size
            anchor_size = all_anchor_size[fg_mask] if cfg.RCNN.SIZE_RES_ON_ROI else MEAN_SIZE

            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(paddle.reshape(rcnn_reg,[batch_size, -1])[fg_mask],
                                        paddle.reshape(gt_boxes3d_ct,[batch_size, 7])[fg_mask],
                                        loc_scope=cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size=anchor_size,
                                        get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine=True)

            loss_size = 3 * loss_size  # consistent with old codes
            rcnn_loss_reg = loss_loc + loss_angle + loss_size
            tb_dict.update(reg_loss_dict)
        else:
            loss_loc = loss_angle = loss_size = rcnn_loss_reg = rcnn_loss_cls * 0

        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict['rcnn_loss_cls'] = rcnn_loss_cls.item(0)
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item(0)
        tb_dict['rcnn_loss'] = rcnn_loss.item(0)

        tb_dict['rcnn_loss_loc'] = loss_loc.item(0)
        tb_dict['rcnn_loss_angle'] = loss_angle.item(0)
        tb_dict['rcnn_loss_size'] = loss_size.item(0)

        tb_dict['rcnn_cls_fg'] = (cls_label > 0).sum().item(0)
        tb_dict['rcnn_cls_bg'] = (cls_label == 0).sum().item(0)
        tb_dict['rcnn_reg_fg'] = reg_valid_mask.sum().item(0)

        return rcnn_loss

    return model_fn
