from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models import BaseDetector
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) /
                     (mask_pred_.sum([1, 2]) + 1e-6))


@MODELS.register_module()
class WaveInst(BaseDetector):
    def __init__(self,
                 data_preprocessor: ConfigType,
                 backbone: ConfigType,
                 dwtbranch: ConfigType,
                 encoder: ConfigType,
                 decoder: ConfigType,
                 criterion: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # backbone
        self.backbone = MODELS.build(backbone)
        # wavelet branch
        self.dwtbranch = MODELS.build(dwtbranch)
        # encoder & decoder
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)

        # matcher & loss (matcher is built in loss)
        self.criterion = MODELS.build(criterion)

        # inference
        self.cls_threshold = test_cfg.score_thr
        self.mask_threshold = test_cfg.mask_thr_binary

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        x = self.backbone(batch_inputs)
        f_spectral = self.dwtbranch(batch_inputs)
        x = self.encoder(x, f_spectral)
        results = self.decoder(x)
        return results

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        max_shape = batch_inputs.shape[-2:]
        output = self._forward(batch_inputs)

        pred_scores = output['pred_logits'].sigmoid()
        pred_masks = output['pred_masks'].sigmoid()
        pred_objectness = output['pred_scores'].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_objectness)

        results_list = []
        for batch_idx, (scores_per_image, mask_pred_per_image,
                        datasample) in enumerate(
            zip(pred_scores, pred_masks, batch_data_samples)):
            result = InstanceData()
            # max/argmax
            scores, labels = scores_per_image.max(dim=-1)
            # cls threshold
            keep = scores > self.cls_threshold
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = mask_pred_per_image[keep]

            if scores.size(0) == 0:
                result.scores = scores
                result.labels = labels
                results_list.append(result)
                continue

            img_meta = datasample.metainfo
            # rescoring mask using maskness
            scores = rescoring_mask(scores,
                                    mask_pred_per_image > self.mask_threshold,
                                    mask_pred_per_image)
            h, w = img_meta['img_shape'][:2]
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image.unsqueeze(1),
                size=max_shape,
                mode='bilinear',
                align_corners=False)[:, :, :h, :w]

            if rescale:
                ori_h, ori_w = img_meta['ori_shape'][:2]
                mask_pred_per_image = F.interpolate(
                    mask_pred_per_image,
                    size=(ori_h, ori_w),
                    mode='bilinear',
                    align_corners=False).squeeze(1)

            mask_pred = mask_pred_per_image > self.mask_threshold
            result.masks = mask_pred
            result.scores = scores
            result.labels = labels
            # create an empty bbox in InstanceData to avoid bugs when
            # calculating metrics.
            result.bboxes = result.scores.new_zeros(len(scores), 4)
            results_list.append(result)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        outs = self._forward(batch_inputs)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = unpack_gt_instances(batch_data_samples)

        losses = self.criterion(outs, batch_gt_instances, batch_img_metas,
                                batch_gt_instances_ignore)
        return losses

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs)
        f_spectral = self.dwtbranch(batch_inputs)
        x = self.encoder(x, f_spectral)
        return x
