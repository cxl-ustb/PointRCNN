import paddle
import paddle.optimizer as optim
import paddle.optimizer.lr as lr_sched
import paddle.nn as nn
from paddle.io import DataLoader
import os
import argparse
import logging
import sys
sys.path.append('/home/cxl/PaperContest/PointRCNN_paddle')
from lib.net.point_rcnn import PointRCNN
import lib.net.train_functions as train_functions
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.config import cfg, cfg_from_file, save_config_to_file
import tools.train_utils.train_utils as train_utils


# --cfg_file cfgs/default.yaml --batch_size 4 --train_mode rcnn --epochs 70  --ckpt_save_interval 2
parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml', help='specify the config for training')
parser.add_argument("--train_mode", type=str, default='rpn', required=True, help="specify the training mode")
parser.add_argument("--batch_size", type=int, default=16, required=True, help="batch size for training")
parser.add_argument("--epochs", type=int, default=200, required=True, help="Number of epochs to train for")

parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
parser.add_argument("--ckpt_save_interval", type=int, default=1, help="number of training epochs")
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')


parser.add_argument("--gt_database", type=str, default='gt_database/train_gt_database_3level_Car.pkl',
                    help='generated gt database for augmentation')
parser.add_argument("--rcnn_training_roi_dir", type=str, default=None,
                    help='specify the saved rois for rcnn training when using rcnn_offline mode')
parser.add_argument("--rcnn_training_feature_dir", type=str, default=None,
                    help='specify the saved features for rcnn training when using rcnn_offline mode')


args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def create_dataloader(logger):
    DATA_PATH = os.path.join('/home/cxl/PaperContest/PointRCNN/', 'data')

    # create dataloader
    train_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TRAIN.SPLIT, mode='TRAIN',
                                 logger=logger,
                                 classes=cfg.CLASSES,
                                 rcnn_training_roi_dir=args.rcnn_training_roi_dir,
                                 rcnn_training_feature_dir=args.rcnn_training_feature_dir,
                                 gt_database_dir=args.gt_database)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.workers, shuffle=True, collate_fn=train_set.collate_batch,
                              drop_last=True)


    test_loader = None
    return train_loader, test_loader


def create_optimizer(model):

    if cfg.TRAIN.OPTIMIZER == 'adam':

        optimizer = optim.Adam(parameters=model.parameters(),learning_rate=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(parameters=model.parameters(),learning_rate=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY
                              )

    else:
        raise NotImplementedError

    return optimizer


def create_scheduler(optimizer, total_steps, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.LR_DECAY
        return max(cur_decay, cfg.TRAIN.LR_CLIP / cfg.TRAIN.LR)

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.BN_DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.BN_DECAY
        return max(cfg.TRAIN.BN_MOMENTUM * cur_decay, cfg.TRAIN.BNM_CLIP)


    lr_scheduler = lr_sched.LambdaDecay(learning_rate=optimizer.get_lr(),lr_lambda=lr_lbmd, last_epoch=last_epoch)

    bnm_scheduler = train_utils.BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return lr_scheduler, bnm_scheduler


if __name__ == "__main__":
    paddle.device.set_device('gpu')
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]


    cfg.RCNN.ENABLED = True
    cfg.RPN.ENABLED =  True
    cfg.RPN.FIXED = False


    if args.output_dir is not None:
        root_result_dir = args.output_dir
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, 'log_train.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')

    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))

    save_config_to_file(cfg, logger=logger)




    # create dataloader & network & optimizer
    train_loader, test_loader = create_dataloader(logger)

    model = PointRCNN(num_classes=train_loader.dataset.num_class, use_xyz=True, mode='TRAIN')
    optimizer = create_optimizer(model)




    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1


    lr_scheduler, bnm_scheduler = create_scheduler(optimizer, total_steps=len(train_loader) * args.epochs,
                                                   last_epoch=last_epoch)


    lr_warmup_scheduler = None

    # start training
    logger.info('**********************Start training**********************')
    ckpt_dir = os.path.join(root_result_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer = train_utils.Trainer(
        model,
        train_functions.model_joint_fn_decorator(),
        optimizer,
        ckpt_dir=ckpt_dir,
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        model_fn_eval=train_functions.model_joint_fn_decorator(),
        eval_frequency=1,
        lr_warmup_scheduler=lr_warmup_scheduler,
        warmup_epoch=cfg.TRAIN.WARMUP_EPOCH,
        grad_norm_clip=cfg.TRAIN.GRAD_NORM_CLIP
    )

    trainer.train(
        it,
        start_epoch,
        args.epochs,
        train_loader,
        test_loader,
        ckpt_save_interval=args.ckpt_save_interval,
        lr_scheduler_each_iter=(cfg.TRAIN.OPTIMIZER == 'adam_onecycle')
    )

    logger.info('**********************End training**********************')
