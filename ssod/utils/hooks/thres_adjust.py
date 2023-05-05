from mmcv.runner.hooks import HOOKS, Hook
from ..logger import get_root_logger
@HOOKS.register_module()
class ThresAdjust(Hook):
    def __init__(self,adjust_step,rpn_decay_factor,rcnn_decay_factor,rcnn_end_step,rpn_end_step):
        self.adjust_step = adjust_step
        self.rcnn_decay_factor = rcnn_decay_factor
        self.rpn_decay_factor = rpn_decay_factor
        self.rpn_end_step =rpn_end_step
        self.rcnn_end_step = rcnn_end_step


    def before_train_iter(self, runner):
        curr_step = runner.iter
        # rpn_thres = runner.model.module.train_cfg.c_pseudo_threshold
        # rcnn_thres = runner.model.module.train_cfg.cls_pseudo_threshold

        if curr_step <= self.rpn_end_step:
            if curr_step != 0 and curr_step % self.adjust_step == 0:
                runner.model.module.train_cfg.rpn_pseudo_threshold -= self.rpn_decay_factor

        if curr_step <= self.rcnn_end_step:
            if curr_step != 0 and curr_step % self.adjust_step == 0:
                runner.model.module.train_cfg.cls_pseudo_threshold -= self.rcnn_decay_factor


        # log 记录
        runner.log_buffer.output['rcnn_pesudo_threshold'] = runner.model.module.train_cfg.cls_pseudo_threshold
        runner.log_buffer.output['rpn_pesudo_threshold'] = runner.model.module.train_cfg.rpn_pseudo_threshold