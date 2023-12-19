import torch
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

class PromptModelPredictor(DetectionPredictor):
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'
    def adjust_bboxes_to_image_border(self, boxes, image_shape, threshold=20):    
        h, w = image_shape
        boxes[:, 0] = torch.where(boxes[:, 0] < threshold, torch.tensor(
            0, dtype=torch.float, device=boxes.device), boxes[:, 0])  # x1
        boxes[:, 1] = torch.where(boxes[:, 1] < threshold, torch.tensor(
            0, dtype=torch.float, device=boxes.device), boxes[:, 1])  # y1
        boxes[:, 2] = torch.where(boxes[:, 2] > w - threshold, torch.tensor(
            w, dtype=torch.float, device=boxes.device), boxes[:, 2])  # x2
        boxes[:, 3] = torch.where(boxes[:, 3] > h - threshold, torch.tensor(
            h, dtype=torch.float, device=boxes.device), boxes[:, 3])  # y2
        return boxes
    def postprocess(self, preds, img, orig_imgs):
        p = ops.non_max_suppression(preds[0],
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    nc=len(self.model.names),
                                    classes=self.args.classes)
        results = []
        if len(p) == 0 or len(p[0]) == 0:
            print("No object detected.")
            return results
        full_box = torch.zeros_like(p[0][0])
        full_box[2], full_box[3], full_box[4], full_box[6:] = img.shape[3], img.shape[2], 1.0, 1.0
        full_box = full_box.view(1, -1)
        self.adjust_bboxes_to_image_border(p[0][:, :4], img.shape[2:]) 
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            if not len(pred): 
                results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6]))
                continue
            if self.args.retina_masks:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            else:
                if not isinstance(orig_imgs, torch.Tensor):
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(
                Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=torch.zeros_like(img)))
        return results
