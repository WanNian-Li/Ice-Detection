"""
models/yolo.py
==============
YOLO 实例分割封装：将 Ultralytics YOLO11-seg 适配为与 Mask R-CNN 相同的接口。

训练模式：forward(images, targets) → Dict[str, Tensor]（损失字典）
推理模式：forward(images)          → List[Dict]（boxes/labels/scores/masks）
"""

import torch
import torch.nn as nn


class YOLOSegWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.utils.loss import v8SegmentationLoss

        yolo_cfg = cfg.model.yolo
        num_classes = int(yolo_cfg.num_classes)

        # 加载预训练模型（首次运行自动下载 ~22 MB）
        _yolo = YOLO(str(yolo_cfg.model))
        inner = _yolo.model  # 底层 nn.Module (SegmentationModel)

        # 若类别数与预训练头不匹配，替换分类卷积层
        detect = inner.model[-1]
        if detect.nc != num_classes:
            _patch_head(detect, num_classes)

        self.inner = inner

        # v8SegmentationLoss 通过属性访问 model.args（需支持 .box .cls .dfl 等），
        # 但从 .pt 加载的 SegmentationModel.args 是普通 dict，且可能缺少训练超参。
        if isinstance(self.inner.args, dict):
            from ultralytics.utils import IterableSimpleNamespace
            _HYPER_DEFAULTS = {
                "box": 7.5, "cls": 0.5, "dfl": 1.5,
                "overlap_mask": True, "mask_ratio": 4,
            }
            self.inner.args = IterableSimpleNamespace(
                **{**_HYPER_DEFAULTS, **self.inner.args}
            )

        # 损失函数必须在 patch 之后初始化，以读取正确的 nc/no
        self.criterion = v8SegmentationLoss(self.inner)

        # 冻结骨干（仅微调检测头）
        if bool(yolo_cfg.freeze_backbone):
            for param in self.inner.model[:-1].parameters():
                param.requires_grad_(False)

        self.score_thresh = float(cfg.evaluate.score_threshold)
        self.nms_thresh = float(cfg.inference.nms_iou_threshold)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)

        # ultralytics 的 stride 是普通 tensor 属性（非 buffer），不会被 super().to() 移动；
        # 需手动迁移 inner 本身及检测头上的 stride。
        for obj in (self.inner, self.inner.model[-1]):
            if hasattr(obj, "stride") and isinstance(obj.stride, torch.Tensor):
                obj.stride = obj.stride.to(*args, **kwargs)

        # v8SegmentationLoss 不是 nn.Module，手动迁移其所有内部组件
        for name, val in list(vars(self.criterion).items()):
            if isinstance(val, nn.Module):
                val.to(*args, **kwargs)
            elif isinstance(val, torch.Tensor):
                setattr(self.criterion, name, val.to(*args, **kwargs))

        # criterion.device 在 init 时记录的是 CPU device，需同步到新设备
        self.criterion.device = next(self.inner.parameters()).device

        return result

    def forward(self, images, targets=None):
        if self.training:
            return self._train_forward(images, targets)
        return self._eval_forward(images)

    # ──────────────────────────────────────────────────────────────
    # 训练前向
    # ──────────────────────────────────────────────────────────────

    def _train_forward(self, images, targets):
        imgs = torch.stack(images)  # (B, 3, H, W)
        batch = _targets_to_yolo_batch(imgs, targets)
        preds = self.inner(imgs)                          # multi-scale + proto
        # criterion 返回 (total_loss_with_grad, detached_items[box, seg, cls])
        # train 循环 sum(loss_dict.values()) 后 backward；
        # 只放 total_loss，避免 detached 分量被加入导致数值翻倍
        total_loss, _ = self.criterion(preds, batch)
        return {"loss_total": total_loss.sum()}

    # ──────────────────────────────────────────────────────────────
    # 推理前向：输出格式与 Mask R-CNN 完全一致
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_pred(H, W, dev):
        return {
            "boxes":  torch.zeros(0, 4, device=dev),
            "labels": torch.zeros(0, device=dev, dtype=torch.int64),
            "scores": torch.zeros(0, device=dev),
            "masks":  torch.zeros(0, 1, H, W, device=dev),
        }

    def _eval_forward(self, images):
        import torch.nn.functional as F
        from torchvision.ops import batched_nms

        imgs = torch.stack(images)
        B, _, H, W = imgs.shape
        nc = self.inner.model[-1].nc
        dev = imgs.device

        with torch.no_grad():
            preds = self.inner(imgs)

        try:
            # YOLO11-seg eval 输出结构（实测确认）：
            #   preds[0][0]: (B, 4+nc+nm, A) — 已解码 xyxy 像素坐标 + sigmoid cls + raw mask coefs
            #   preds[0][1]: (B, nm, mh, mw) — proto masks
            pred_det = preds[0][0]             # (B, 4+nc+nm, A)
            proto    = preds[0][1]             # (B, nm, mh, mw)
            nm = proto.shape[1]

            outputs = []
            for b in range(B):
                pd = pred_det[b]               # (4+nc+nm, A)

                # boxes 已是 xyxy 像素坐标（Detect.forward 中 dfl+dist2bbox 解码）
                boxes_raw  = pd[:4].T          # (A, 4) xyxy
                cls_scores = pd[4:4 + nc].T    # (A, nc) 已 sigmoid
                coefs      = pd[4 + nc:].T     # (A, nm) raw

                scores, cls_ids = cls_scores.max(1)
                # AP50 需要在所有置信度阈值上积分，必须用极低阈值保留所有候选；
                # 0.3 高阈值只适合推理可视化，不能用于验证阶段
                keep = scores >= 0.01
                if keep.sum() == 0:
                    outputs.append(self._empty_pred(H, W, dev))
                    continue

                boxes_raw = boxes_raw[keep]
                scores    = scores[keep]
                cls_ids   = cls_ids[keep]
                coefs     = coefs[keep]

                # 限制 NMS 前候选框数量，防止 mask 生成 O(n×H×W) 爆炸
                # 训练几轮后模型对大量锚点置信度 >0.01，不限制会产生 GB 级中间张量
                MAX_PRE_NMS = 300
                if len(scores) > MAX_PRE_NMS:
                    topk = torch.topk(scores, MAX_PRE_NMS).indices
                    boxes_raw = boxes_raw[topk]
                    scores    = scores[topk]
                    cls_ids   = cls_ids[topk]
                    coefs     = coefs[topk]

                # xyxy 坐标裁剪到图像范围
                x1 = boxes_raw[:, 0].clamp(0, W)
                y1 = boxes_raw[:, 1].clamp(0, H)
                x2 = boxes_raw[:, 2].clamp(0, W)
                y2 = boxes_raw[:, 3].clamp(0, H)
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

                idx = batched_nms(boxes_xyxy, scores, cls_ids, self.nms_thresh)
                boxes_xyxy = boxes_xyxy[idx]
                scores     = scores[idx]
                cls_ids    = cls_ids[idx]
                coefs      = coefs[idx]

                # post-NMS 上限：100 个预测足够 AP50 曲线积分；
                # 超出部分是低置信度 FP，移除后 AP 微幅高估但训练趋势监控不受影响
                MAX_DETS = 100
                if len(scores) > MAX_DETS:
                    topk = torch.topk(scores, MAX_DETS).indices
                    boxes_xyxy = boxes_xyxy[topk]
                    scores     = scores[topk]
                    cls_ids    = cls_ids[topk]
                    coefs      = coefs[topk]

                pb  = proto[b]                         # (nm, mh, mw)
                mh, mw = pb.shape[-2:]
                n = len(boxes_xyxy)
                masks = torch.sigmoid(coefs @ pb.view(nm, -1)).view(n, mh, mw)
                masks = F.interpolate(
                    masks.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
                )[0]

                outputs.append({
                    "boxes":  boxes_xyxy,
                    "labels": cls_ids.long() + 1,
                    "scores": scores,
                    "masks":  (masks > 0.5).unsqueeze(1).float(),
                })

            return outputs

        except Exception as e:
            print(f"[YOLO EVAL] 解析失败: {e}，本批次返回空预测（训练继续）", flush=True)
            import traceback; traceback.print_exc()
            return [self._empty_pred(H, W, dev)] * B


# ──────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────

def _find_proto(obj, batch_size: int):
    """从 Segment head 的第二返回值中递归找出 proto 张量 (B, nm, mh, mw)。
    兼容 ultralytics 各版本返回 tensor / tuple / list / dict 的情况。"""
    import torch
    if isinstance(obj, torch.Tensor) and obj.dim() == 4 and obj.shape[0] == batch_size:
        return obj
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (tuple, list)):
        for item in obj:
            result = _find_proto(item, batch_size)
            if result is not None:
                return result
    return None


def _patch_head(detect, num_classes: int) -> None:
    """替换 Detect/Segment 头的分类卷积层以适配自定义类别数。"""
    if not hasattr(detect, "cv3"):
        raise RuntimeError(
            f"无法识别检测头结构（不含 cv3），当前层: {type(detect).__name__}。"
            "请确认使用的是 YOLOv8/YOLO11 分割模型。"
        )

    for branch in detect.cv3:
        last = branch[-1]                     # 最后一层 Conv2d
        new_conv = nn.Conv2d(last.in_channels, num_classes, 1)
        nn.init.normal_(new_conv.weight, std=0.01)
        nn.init.constant_(new_conv.bias, -4.0)  # 低先验，适合稀疏目标
        branch[-1] = new_conv

    detect.nc = num_classes
    detect.no = num_classes + detect.reg_max * 4  # DFL: reg_max=16 → +64


def _targets_to_yolo_batch(imgs: torch.Tensor, targets: list) -> dict:
    """
    将 Mask R-CNN 格式 targets 转换为 v8SegmentationLoss 所需的 batch dict。

    targets[i] 含:
        boxes  (N, 4) xyxy 绝对坐标
        labels (N,)   1-indexed
        masks  (N, H, W) bool

    输出 batch:
        img       (B, 3, H, W)
        cls       (N_total, 1) float 0-indexed
        bboxes    (N_total, 4) xywh 归一化
        masks     (N_total, H, W) float [0,1]
        batch_idx (N_total,) long
    """
    B, _C, H, W = imgs.shape
    device = imgs.device

    cls_list, bbox_list, mask_list, bidx_list = [], [], [], []

    for i, t in enumerate(targets):
        n = len(t["labels"])
        if n == 0:
            continue

        cls_list.append((t["labels"].float() - 1).unsqueeze(1))  # 1→0 indexed

        b = t["boxes"].float()
        cx = (b[:, 0] + b[:, 2]) / 2 / W
        cy = (b[:, 1] + b[:, 3]) / 2 / H
        bw = (b[:, 2] - b[:, 0]) / W
        bh = (b[:, 3] - b[:, 1]) / H
        bbox_list.append(torch.stack([cx, cy, bw, bh], dim=1))

        mask_list.append(t["masks"].float())  # bool → float [0,1]
        bidx_list.append(torch.full((n,), i, device=device, dtype=torch.long))

    if cls_list:
        return {
            "img":       imgs,
            "cls":       torch.cat(cls_list).to(device),
            "bboxes":    torch.cat(bbox_list).to(device),
            "masks":     torch.cat(mask_list).to(device),
            "batch_idx": torch.cat(bidx_list),
        }

    return {
        "img":       imgs,
        "cls":       torch.zeros(0, 1, device=device),
        "bboxes":    torch.zeros(0, 4, device=device),
        "masks":     torch.zeros(0, H, W, device=device),
        "batch_idx": torch.zeros(0, dtype=torch.long, device=device),
    }


def build_yolo(cfg) -> YOLOSegWrapper:
    return YOLOSegWrapper(cfg)


def get_yolo_params(model: YOLOSegWrapper, lr: float, weight_decay: float) -> list:
    """
    差异化学习率参数组：
      - 骨干（冻结时为空）：lr × 0.1
      - 检测头：lr
    """
    head_ids = {id(p) for p in model.inner.model[-1].parameters()}

    head_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) in head_ids]
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in head_ids]

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": lr * 0.1, "weight_decay": weight_decay})
    groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
    return groups
