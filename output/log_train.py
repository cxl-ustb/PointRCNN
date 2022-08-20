import paddle
import numpy as np
import sys
sys.path.append('/home/cxl/PaperContest/PointRCNN_paddle')
from lib.utils.torch2paddle_utils import paddle_gather


def rotate_pc_along_paddle(pc,rot_angle):
    '''
    :param pc: (N,3+C)
    :param rot_angle: (N)
    :return:
    '''
    cosa=paddle.reshape(paddle.cos(rot_angle),[-1,1])
    sina=paddle.reshape(paddle.sin(rot_angle),[-1,1])

    raw_1=paddle.concat([cosa,-sina],axis=1)
    raw_2=paddle.concat([sina,cosa],axis=1)
    R=paddle.concat((paddle.unsqueeze(raw_1,axis=1),paddle.unsqueeze(raw_2,axis=1)),axis=1)

    pc_temp=paddle.unsqueeze(pc[:,[0,2]],axis=1)
    pc[:,0:2]=paddle.squeeze(paddle.matmul(pc_temp,paddle.transpose(R,[0,2,1])),axis=1)
    return pc

def decode_bbox_target(roi_box3d,pred_reg,loc_scope,loc_bin_size,num_head_bin,anchor_size,
                       get_xz_fine=True,get_y_by_bin=False,loc_y_scope=0.5,loc_y_bin_size=0.25,get_ry_fine=False):
    '''
    :param roi_box3d: (N,7) (16384,3)
    :param pred_reg:   torch.Size([16384, 52])
    :param loc_scope:  3.0
    :param loc_bin_size: 0.5
    :param num_head_bin: 12
    :param anchor_size:  tensor([1.5256, 1.6286, 3.8831], device='cuda:0')
    :param get_xz_fine:
    :param get_y_by_bin:
    :param loc_y_scope: 0.5
    :param loc_y_bin_size: 0.25
    :param get_ry_fine:
    :return:
    '''
    paddle.device.set_device('gpu')
    # roi_box3d roi的点的x,y,z坐标
    # per_loc_bin_num 12
    per_loc_bin_num=int(loc_scope/loc_bin_size)*2
    # loc_y_bin_num 4
    loc_y_bin_num=int(loc_y_scope/loc_y_bin_size)*2

    # recover xz localization
    # pred_reg[0:12] x_bin
    x_bin_l,x_bin_r=0,per_loc_bin_num
    # pred_reg[12:24] z_bin
    z_bin_l,z_bin_r=per_loc_bin_num,per_loc_bin_num*2
    start_offset=z_bin_r
    # 将x,z都分成了12个bin，选择出中心点落在其中概率最大的那个bin的索引（x,z)
    x_bin=paddle.argmax(pred_reg[:,x_bin_l:x_bin_r],axis=1)
    z_bin=paddle.argmax(pred_reg[:,z_bin_l:z_bin_r],axis=1)
    # 计算出以roi3d的中心点，也就是以bounding box的中心点为坐标原点时，预测出来的前景物体的中心点坐标，即canonical坐标系下的坐标为（pos_x,pos_z)
    pos_x=x_bin.astype('float32')*loc_bin_size+loc_bin_size/2-loc_scope
    pos_z=z_bin.astype('float32')*loc_bin_size+loc_bin_size/2-loc_scope

    if get_xz_fine:
        # pred_reg[24:32] x_res
        x_res_l,x_res_r=per_loc_bin_num*2,per_loc_bin_num*3
        # pred_reg[32:48] z_res
        z_res_l,z_res_r=per_loc_bin_num*3,per_loc_bin_num*4
        start_offset=z_res_r
        # 根据 x_bin的索引获得对应的x_res索引
        x_res_norm=paddle.squeeze(paddle_gather(pred_reg[:,x_res_l:x_res_r],index=paddle.unsqueeze(x_bin,axis=1)),
                                  axis=1)
        # 根据 z_bin的索引获得对应的z_res索引
        z_res_norm = paddle.squeeze(paddle_gather(pred_reg[:, z_res_l:z_res_r], index=paddle.unsqueeze(z_bin, axis=1)),
                                  axis=1)
        x_res=x_res_norm*loc_bin_size
        z_res=z_res_norm*loc_bin_size

        pos_x+=x_res
        pos_z+=z_res

        # recover y localization
        if get_y_by_bin:
            y_bin_l,y_bin_r=start_offset,start_offset+loc_y_bin_num
            y_res_l,y_res_r=y_bin_r,y_bin_r+loc_y_bin_num
            start_offset=y_res_r

            y_bin=paddle.argmax(pred_reg[:,y_bin_l:y_bin_r],axis=1)
            y_res_norm=paddle.squeeze(paddle.gather(pred_reg[:,y_res_l:y_res_r],axis=1,index=paddle.unsqueeze(y_bin,axis=1)),axis=1)
            y_res=y_res_norm*loc_y_bin_size
            pos_y=y_bin.astype('float32')*loc_y_bin_size+loc_y_bin_size/2-loc_y_scope+y_res
            pos_y=pos_y+roi_box3d[:,1]
        else:
            y_offset_l,y_offset_r=start_offset,start_offset+1
            start_offset=y_offset_r
            # bounding box中心点坐标+预测出来的y的偏移，得到预测出来的前景物体的中心店y的值
            pos_y=roi_box3d[:,1]+pred_reg[:,y_offset_l]

        #recover ry rotation
        ry_bin_l,ry_bin_r=start_offset,start_offset+num_head_bin
        ry_res_l,ry_res_r=ry_bin_r,ry_bin_r+num_head_bin

        ry_bin=paddle.argmax(pred_reg[:,ry_bin_l:ry_bin_r],axis=1)
        ry_res_norm=paddle.squeeze(paddle_gather(pred_reg[:,ry_res_l:ry_res_r],axis=1,index=paddle.unsqueeze(ry_bin,axis=1)),axis=1)

        if get_ry_fine:
            # divide pi/2 to several bins
            angle_per_class=(np.pi/2)/num_head_bin
            ry_res=ry_res_norm*(angle_per_class/2)
            ry=(ry_bin.astype('float32')*angle_per_class+angle_per_class/2)+ry_res-np.pi/4
        else:
            angle_per_class=(2*np.pi)/num_head_bin
            ry_res=ry_res_norm*(angle_per_class/2)

            # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
            ry = (ry_bin.astype('float32') * angle_per_class + ry_res) % (2 * np.pi)
            ry[ry > np.pi] -= 2 * np.pi

        #recover size
        size_res_l,size_res_r=ry_res_r,ry_res_r+3
        assert size_res_r==pred_reg.shape[1]

        size_res_norm=pred_reg[:,size_res_l:size_res_r]
        hwl=size_res_norm*anchor_size+anchor_size

        #shift to original coords
        roi_center=roi_box3d[:,0:3]
        shift_ret_box3d=paddle.concat((paddle.reshape(pos_x,[-1,1]),
                                       paddle.reshape(pos_y,[-1,1]),
                                       paddle.reshape(pos_z,[-1,1]),
                                       hwl,
                                       paddle.reshape(ry,[-1,1])),axis=1)

        ret_box3d=shift_ret_box3d
        if roi_box3d.shape[1]==7:
            roi_ry=roi_box3d[:,6]
            ret_box3d=rotate_pc_along_paddle(shift_ret_box3d,-roi_ry)
            ret_box3d[:,6]+roi_ry

        ret_box3d[:,[0,2]]+=roi_center[:,[0,2]]

        return ret_box3d



