# default_scope = 'mmdet'
_base_ = '../../mmdetection/configs/dcn/mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco.py'
# _base_ = '/home/fpares/prima_projects/delorian_claim/the_delorian_claim_ml/mmdetection/configs/dcn/mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco.py'


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

# Modify dataset related settings
data_root = '/home/fpares/prima_projects/delorian_claim/the_delorian_claim_ml/small_data/CarDD_COCO/'

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=[(1333, 640), (1333, 1200)],
        keep_ratio=False,
        backend='pillow'
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='RandomResize',
        scale=[(1333, 640), (1333, 1200)],
        keep_ratio=False,
        backend='pillow'
    ),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json'
)
test_evaluator = val_evaluator


train_cfg = dict(
    type='EpochBasedTrainLoop',
    val_interval=1
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
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/home/fpares/prima_projects/delorian_claim/the_delorian_claim_ml/model/pretrained/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth'
