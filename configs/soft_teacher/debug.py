_base_ = "base.py"
data_root = '../data/coco/'
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="../data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="../data/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="../data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="../data/coco/train2017/",
        ),
    ),
    val=dict(
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        ),
    test=dict(
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        ),
    sampler=dict(
        train=dict(
            _delete_ =True,
            type = 'GroupSampler'
            # sample_ratio=[1, 4],
        )
    ),
)

fold = 1
percent = 10

work_dir = "work_dirs/debug/${cfg_name}/"


log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
                name="${cfg_name}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
load_from = "/home/lab/YYF/ssod/SoftTeacher/work_dirs/faster_rcnn_r50_caffe_fpn_coco_partial_180k/10/1/latest.pth"