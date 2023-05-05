_base_ = "base.py"
fold = 2
percent = 10
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
        img_prefix="data/coco/train2017/",
    ),
)
work_dir = "work_dirs/base_line_for_analyze/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=100,
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
