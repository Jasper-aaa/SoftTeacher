_base_ = "base.py"
model = dict(
    train_cfg=dict(
        rpn=dict(
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.4),
                min_bbox_size=0),
        )
    ),
    test_cfg = dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.5,min_score = 0.01),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5,min_score = 0.01),
            max_per_img=100)
    )
)

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="data/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="data/coco/train2017/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

fold = 1
percent = 10

semi_wrapper = dict(
    type="SoftTeacher",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=True,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.5,
        cls_pseudo_threshold=0.5,
        reg_pseudo_threshold=0.02,
        jitter_times=1,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=2.0,
    ),
    test_cfg=dict(inference_on="student"),
)

work_dir = "work_dirs/used_pretrained/new/scorethres/0.5/${cfg_name}/${percent}/${fold}"
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

load_from = f"/home/lab/YYF/ssod/SoftTeacher/work_dirs/faster_rcnn_r50_caffe_fpn_coco_partial_180k/{percent}/{fold}/latest.pth"