_base_ = '../dcn/mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco.py'

classes = ('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat')
n_classes = len(classes)

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        stage_with_dcn=(True, True, True, True)),
    roi_head=dict(
        bbox_head=dict(
            num_classes=n_classes,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.5,
                loss_weight=1.0)
            ),
        mask_head=dict(num_classes=n_classes)))


data_root = '/home/fpares/prima_projects/delorian_claim/the_delorian_claim_ml/data/CarDD_COCO'

train_pipeline = [
    dict(
        type='Resize',
        scale=[(1333, 640), (1333, 1200)],
        multiscale_mode='range',
        keep_ratio=False,
        backend='pillow'
    ),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(
        type='Resize',
        scale=[(1333, 640), (1333, 1200)],
        multiscale_mode='range',
        keep_ratio=False,
        backend='pillow'
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    val_interval=6
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer = dict(lr=0.005)  # LR
)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=500),
    checkpoint=dict(type='CheckpointHook', interval=6)
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '../../model/pretrained/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth'
