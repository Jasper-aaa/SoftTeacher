


import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

# Choose to use a config and initialize the detector
config = "/home/lab/YYF/ssod/SoftTeacher/configs/baseline/faster_rcnn_r50_caffe_fpn_coco_partial_180k.py"
# Setup a checkpoint file to load
checkpoint = "/home/lab/YYF/ssod/SoftTeacher/work_dirs/faster_rcnn_r50_caffe_fpn_coco_partial_180k/10/1/latest.pth"
out_dir = "/home/lab/YYF/ssod/SoftTeacher/tools/plot/out_put"
# Set the device to be used for evaluation
device='cuda:0'
iou_thres = 0.5
# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None
config.model.test_cfg.rpn.nms.iou_threshold = iou_thres
config.model.test_cfg.rcnn.nms.iou_threshold = iou_thres
# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()

img = '000000541123.jpg'
result = inference_detector(model, img)

show_result_pyplot(model, img, result, score_thr=0.9,out_file="result/0.9_1123.jpg")