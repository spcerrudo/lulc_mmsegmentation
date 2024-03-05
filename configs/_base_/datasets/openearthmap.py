# dataset settings
dataset_type = 'OpenEarthMapDataset'
data_root = r'/home/sam_cerrudo/Codes/LULC_Mapping/OpenEarthMap1000'

# data augmentation settings
albu_train_transforms = [
    dict(type='RandomBrightnessContrast', p=0.2),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5)
]

# pipeline for training
train_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Albu', transforms=albu_train_transforms),
    dict(type='PackSegInputs')
]

# pipeline for testing
test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# pipeline for test-time augmentation
tta_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[[dict(type='LoadAnnotations')],
                    [dict(type='PackSegInputs')]])
]

# train dataloader settings
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/images', seg_map_path='train/labels'),
        pipeline=train_pipeline))

# val dataloader settings
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/images', seg_map_path='val/labels'),
        pipeline=test_pipeline))

# test dataloader settings
test_dataloader = val_dataloader

# evaluator settings
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator