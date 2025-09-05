import copy
from torch.nn import functional as F
import torch
import numpy
from propvg.core.structure.boxes import Boxes
from propvg.core.structure.instances import Instances
from propvg.core.structure.postprocessing import detector_postprocess
from propvg.layers.box_ops import box_cxcywh_to_xyxy
from propvg.models import MODELS
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils

from propvg.models.postprocess.nms import nms
from .one_stage import OneStageModel
import numpy as np
from PIL import Image, ImageDraw
from propvg.utils import is_main
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


colors = [
    (237, 123, 187),
    (204, 222, 58),
    (119, 194, 221),
    (224, 138, 51),
    (134, 208, 69),
    (191, 56, 154),
    (128, 154, 238),
    (245, 207, 103),
    (106, 133, 202),
    (212, 175, 54),
    (171, 220, 249),
    (153, 216, 191),
    (243, 114, 176),
    (102, 194, 244),
    (255, 165, 109),
    (160, 160, 160),
    (173, 216, 230),
    (147, 112, 219),
    (255, 215, 0),
    (135, 206, 235),
    (255, 127, 80),
    (220, 20, 60),
    (139, 0, 139),
    (165, 42, 42),
    (255, 99, 71),
    (127, 255, 212),
    (124, 252, 0),
    (255, 20, 147),
    (255, 250, 240),
    (152, 251, 152),
    (75, 0, 130),
    (255, 228, 196),
    (70, 130, 180),
    (255, 69, 0),
    (46, 139, 87),
    (255, 222, 173),
    (218, 165, 32),
    (0, 191, 255),
    (238, 130, 238),
    (244, 164, 96),
    (139, 69, 19),
    (0, 128, 128),
    (255, 105, 180),
    (128, 128, 0),
    (255, 192, 203),
    (221, 160, 221),
    (250, 128, 114),
    (34, 139, 34),
    (255, 239, 213),
    (100, 149, 237),
    (255, 160, 122),
    (222, 184, 135),
    (32, 178, 170),
    (210, 105, 30),
    (72, 61, 139),
    (0, 250, 154),
    (186, 85, 211),
    (255, 218, 185),
    (176, 224, 230),
]


@MODELS.register_module()
class MIXGrefUniModel_OMG(OneStageModel):
    def __init__(
        self,
        word_emb,
        num_token,
        vis_enc,
        lan_enc,
        head,
        fusion,
        mask_save_target_dir="",
        process_visual=True,
        post_params={"score_weighted": True, "mask_threshold": 0.5, "score_threshold": 0.7, "with_nms": False},
        visualize_params={"row_columns": (2, 5)},
        visual_mode="val",
    ):
        super(MIXGrefUniModel_OMG, self).__init__(word_emb, num_token, vis_enc, lan_enc, head, fusion)
        self.patch_size = vis_enc["patch_size"]
        self.visualize = process_visual
        if is_main() and self.visualize:
            self.train_mask_save_target_dir = os.path.join(mask_save_target_dir, "train_vis")
            self.val_mask_save_target_dir = os.path.join(mask_save_target_dir, "val_vis")
            self.test_mask_save_target_dir = os.path.join(mask_save_target_dir, "test_vis")
            os.makedirs(self.train_mask_save_target_dir, exist_ok=True)
            os.makedirs(self.val_mask_save_target_dir, exist_ok=True)
            os.makedirs(self.test_mask_save_target_dir, exist_ok=True)
        self.iter = 0
        self.threshold = post_params["mask_threshold"]
        self.box_threshold = post_params["score_threshold"]
        self.score_weighted = post_params["score_weighted"]
        self.with_nms = post_params["with_nms"]
        self.visualize_params = visualize_params
        self.visual_mode = visual_mode

    def forward_train(
        self,
        img,
        ref_expr_inds,
        img_metas,
        text_attention_mask=None,
        gt_bbox=None,
        gt_mask_rle=None,
        gt_mask_parts_rle=None,
        rescale=False,
        epoch=None,
    ):
        """Args:
        img (tensor): [batch_size, c, h_batch, w_batch].

        ref_expr_inds (tensor): [batch_size, max_token].

        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `seqtr/datasets/pipelines/formatting.py:CollectData`.

        gt_bbox (list[tensor]): [4, ], in [tl_x, tl_y, br_x, br_y] format,
            the coordinates are in 'img_shape' scale.

        gt_mask_vertices (list[tensor]): [batch_size, 2, num_ray], padded values are -1,
            the coordinates are in 'pad_shape' scale.

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.

        """
        B, _, H, W = img.shape
        img_feat, text_feat, cls_feat = self.extract_visual_language(img, ref_expr_inds, text_attention_mask)
        img_feat = img_feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)  # (B, C, H, W)

        targets = {
            "mask": gt_mask_rle,
            "bbox": gt_bbox,
            "img_metas": img_metas,
            "epoch": epoch,
            "mask_parts": gt_mask_parts_rle,
        }

        losses_dict, pred_dict, extra_dict = self.head.forward_train(
            img_feat, copy.deepcopy(targets), cls_feat, text_feat, text_attention_mask, img
        )

        with torch.no_grad():
            predictions = self.get_predictions_parts(
                pred_dict, img_metas, rescale=rescale, with_bbox=True, with_mask=True
            )
        self.iter += 1
        if is_main() and self.iter % 100 == 0 and self.visualize:
            self.visualiation_parts(
                predictions["parts_list"],
                img_metas,
                targets,
                self.train_mask_save_target_dir,
                extra_dict,
                text_attention_mask,
            )

        return losses_dict, _

    def extract_visual_language(self, img, ref_expr_inds, text_attention_mask=None):
        x, y, c = self.vis_enc(img, ref_expr_inds, text_attention_mask)
        return x, y, c

    @torch.no_grad()
    def forward_test(
        self,
        img,
        ref_expr_inds,
        img_metas,
        text_attention_mask=None,
        with_bbox=False,
        with_mask=False,
        gt_bbox=None,
        gt_mask=None,
        gt_mask_parts_rle=None,
        rescale=False,
    ):
        """Args:
        img (tensor): [batch_size, c, h_batch, w_batch].

        ref_expr_inds (tensor): [batch_size, max_token], padded value is 0.

        img_metas (list[dict]): list of image info dict where each dict
            has: 'img_shape', 'scale_factor', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `rec/datasets/pipelines/formatting.py:CollectData`.

        with_bbox/with_mask: whether to generate bbox coordinates or mask contour vertices,
            which has slight differences.

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """
        self.iter += 1

        B, _, H, W = img.shape
        img_feat, text_feat, cls_feat = self.extract_visual_language(img, ref_expr_inds, text_attention_mask)
        img_feat = img_feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)

        targets = {
            "mask": gt_mask,
            "bbox": gt_bbox,
            "img_metas": img_metas,
            "mask_parts": gt_mask_parts_rle,
        }

        pred_dict, extra_dict = self.head.forward_test(img_feat, cls_feat, text_feat, text_attention_mask, img)

        predictions = self.get_predictions_parts(
            pred_dict,
            img_metas,
            rescale=rescale,
            with_bbox=with_bbox,
            with_mask=with_mask,
        )
        if is_main() and self.iter % 2 == 0 and self.visualize:
            try:
                self.visualiation_parts(
                    predictions["parts_list"],
                    img_metas,
                    targets,
                    self.val_mask_save_target_dir if self.visual_mode == "val" else self.test_mask_save_target_dir,
                    extra_dict,
                    text_attention_mask=text_attention_mask,
                )
            except Exception as e:
                print(e)

        return predictions

    def get_predictions_parts(self, pred, img_metas, rescale=False, with_bbox=False, with_mask=False):
        """Args:
        seq_out_dict (dict[tensor]): [batch_size, 4/2*num_ray+1].

        rescale (bool): whether to rescale predictions from `img_shape`/`pad_shape`
            back to `ori_shape`.
        """
        pred_bboxes, pred_masks = [], []
        all_bbox_cls, all_bboxes, nt_labels, global_seg_mask, refer_pred, att_maps = (
            pred.get("pred_class_allobj", None),
            pred.get("pred_bbox_allobj", None),
            pred.get("nt_label", None),
            pred.get("pred_global_mask", None),
            pred.get("refer_label", None),
            pred.get("att_map", None),
        )
        scores, nms_indices = [], []
        parts_list = {
            "pred_box_parts": [],
            "pred_scores": [],
            "pred_mask": [],
            "det_scores": [],
            "refer_scores": [],
            "att_map": [],
        }

        bboxes = all_bboxes
        bbox_cls = all_bbox_cls

        if bboxes is not None and with_bbox:
            image_sizes = [img_meta["img_shape"] for img_meta in img_metas]
            results = self.inference(bbox_cls, bboxes, image_sizes)

            for ind, (results_per_image, img_meta, att_map) in enumerate(zip(results, img_metas, att_maps)):
                pred_nt = 1 - nt_labels[ind]
                image_size = img_meta["img_shape"]
                height = image_size[0]
                width = image_size[1]
                r = detector_postprocess(results_per_image, height, width)
                # infomation extract
                pred_box = r.pred_boxes.tensor
                det_score = r.scores
                # for GTMHead
                refer_score = refer_pred[ind][:, 0]
                score = refer_score

                if self.with_nms:
                    filtered_boxes = copy.deepcopy(pred_box)
                    filtered_scores = copy.deepcopy(score)
                    filtered_indices = nms(filtered_boxes, filtered_scores, 0.7)
                    filtered_boxes = filtered_boxes[filtered_indices]
                    filtered_scores = filtered_scores[filtered_indices]
                    nms_indices.append(filtered_indices)

                if rescale:
                    scale_factors = img_meta["scale_factor"]
                    pred_box /= pred_box.new_tensor(scale_factors)
                if self.with_nms:
                    cur_predict_dict = {
                        "boxes": pred_box,
                        "scores": score,
                        "filtered_boxes": filtered_boxes,
                        "filtered_scores": filtered_scores,
                        "pred_nt": pred_nt,
                    }
                else:
                    cur_predict_dict = {"boxes": pred_box, "scores": score, "pred_nt": pred_nt}
                parts_list["pred_scores"].append(score.cpu().detach().numpy())
                parts_list["pred_box_parts"].append(pred_box.cpu().detach().numpy())
                parts_list["det_scores"].append(det_score.cpu().detach().numpy())
                parts_list["refer_scores"].append(refer_score.cpu().detach().numpy())
                parts_list["att_map"].append(att_map.cpu().detach().numpy())
                scores.append(score)
                pred_bboxes.append(cur_predict_dict)

        if with_mask:

            global_mask_binary = global_seg_mask.sigmoid()
            global_mask_binary[global_mask_binary < self.threshold] = 0.0
            global_mask_binary[global_mask_binary >= self.threshold] = 1.0
            for ind, (img_meta, global_mask) in enumerate(zip(img_metas, global_mask_binary)):
                pred_nt = 1 - nt_labels[ind]
                h_pad, w_pad = img_meta["pad_shape"][:2]
                cur_scores = scores[ind]
                mask_ = torch.any(global_mask, dim=0).int()
                pred_rle = maskUtils.encode(numpy.asfortranarray(mask_.cpu().numpy().astype(np.uint8)))

                if rescale:
                    h_img, w_img = img_meta["ori_shape"][:2]
                    pred_mask = BitmapMasks(maskUtils.decode(pred_rle)[None], h_pad, w_pad)
                    pred_mask = pred_mask.resize((h_img, w_img))
                    pred_mask = pred_mask.masks[0]
                    pred_mask = numpy.asfortranarray(pred_mask)
                    pred_rle = maskUtils.encode(pred_mask)  # dict

                gt_nt = img_meta["empty"]
                pred_masks.append({"pred_masks": pred_rle, "pred_nt": pred_nt, "gt_nt": gt_nt})
                parts_list["pred_mask"].append(mask_.cpu().detach().numpy())

        return dict(pred_bboxes=pred_bboxes, pred_masks=pred_masks, parts_list=parts_list)

    def inference(self, box_cls, box_pred, image_sizes):
        """Inference function for DETR

        Args:
            box_cls (torch.Tensor): tensor of shape ``(batch_size, num_queries, K)``.
                The tensor predicts the classification probability for each query.
            box_pred (torch.Tensor): tensors of shape ``(batch_size, num_queries, 4)``.
                The tensor predicts 4-vector ``(x, y, w, h)`` box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (
            scores_per_image,
            labels_per_image,
            box_pred_per_image,
            image_size,
        ) in enumerate(zip(scores, labels, box_pred, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def visualiation_parts(self, pred_dict, img_metas, targets, save_target_dir, extra_dict, text_attention_mask):
        index, min = 0, 50
        for i in range(text_attention_mask.size(0)):
            value = torch.sum(text_attention_mask[i])
            if value < min:
                min = value
                index = i

        save_filename = os.path.join(save_target_dir, str(self.iter))
        pred_box_parts = pred_dict["pred_box_parts"][index]
        pred_scores = pred_dict["pred_scores"][index]
        pred_det_scores = pred_dict["det_scores"][index]
        pred_refer_scores = pred_dict["refer_scores"][index]
        gt_mask = maskUtils.decode(targets["mask"][index])
        pred_mask = pred_dict["pred_mask"][index]
        gt_all_box = targets["bbox"][index]
        refer_index = img_metas[index]["refer_target_index"]
        gt_box = gt_all_box[refer_index]

        if "img_selected_points" in extra_dict:
            img_selected_points = extra_dict["img_selected_points"][0]
        if "attn_map_query" in extra_dict:
            attn_map = extra_dict["attn_map_query"][index]

        expression = img_metas[index]["expression"]
        file_name = img_metas[index]["filename"]

        row_columns = self.visualize_params["row_columns"]
        # 创建一个新图像
        fig, axs = plt.subplots(
            row_columns[0],
            row_columns[1],
            figsize=(row_columns[1] * 3, row_columns[0] * 3),
        )

        H, W = gt_mask.shape
        img = Image.open(file_name)
        # img = img.convert('L')
        img = img.convert("RGB")
        img = img.resize((W, H))
        for i in range(row_columns[0]):
            for j in range(row_columns[1]):
                # axs[i, j].imshow(img, cmap="gray")
                axs[i, j].imshow(img)
                axs[i, j].axis("off")

        # 对每一行进行水平拼接
        for i in range(row_columns[0]):
            for j in range(row_columns[1]):
                idx = i * row_columns[1] + j
                box = pred_box_parts[idx]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                axs[i, j].add_patch(rect)
                if pred_refer_scores[idx] > 0.5 or pred_det_scores[idx] > 0.5 or pred_scores[idx] > 0.3:
                    if pred_scores[idx] > 0.3:
                        color = "red"
                    elif pred_refer_scores[idx] > 0.5:
                        color = "blue"
                    elif pred_det_scores[idx] > 0.5:
                        color = "green"

                    score_text = f"{pred_scores[idx]:.2f}|D{pred_det_scores[idx]:.2f}|R{pred_refer_scores[idx]:.2f}"
                    axs[i, j].text(
                        box[0] - 30,
                        box[1] - 10,
                        score_text,
                        color=color,
                        fontsize=10,
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                    )

                if "attn_map_query" in extra_dict:
                    tokenized_words = img_metas[index]["tokenized_words"]
                    text_attn_map = attn_map[idx]
                    assert len(tokenized_words) == (50 - min) - 2
                    if len(tokenized_words) < len(text_attn_map) - 2:
                        targeted_text_attn_map = text_attn_map[1 : len(tokenized_words) + 1]
                    else:
                        targeted_text_attn_map = text_attn_map[1:-1]
                    targeted_word = tokenized_words[torch.argmax(targeted_text_attn_map)]
                    axs[i, j].text(
                        0.5,
                        -0.1,
                        targeted_word,  # 显示在图片上方
                        color="blue",
                        fontsize=12,
                        fontweight="bold",
                        verticalalignment="top",
                        horizontalalignment="center",
                        transform=axs[i, j].transAxes,  # 使用 Axes 的坐标系
                    )

        save_filename_query = save_filename + "-{}-querymap.jpg".format(expression)
        plt.savefig(save_filename_query)

        box_gt = (gt_box.cpu().detach().numpy()).astype(np.int32)
        mask_gt = gt_mask.astype(np.int32)
        mask_gt = Image.fromarray(mask_gt * 255)
        image_gt = Image.new("RGB", (W, H))
        image_gt.paste(mask_gt)
        draw_gt = ImageDraw.Draw(image_gt)
        for box in box_gt:
            draw_gt.rectangle(list(box), outline="red", width=2)

        filterd_pred_box = pred_box_parts[np.where(pred_scores > self.box_threshold)]
        box_pred = filterd_pred_box.astype(np.int32)
        mask_pred = pred_mask.astype(np.int32)
        mask_pred = Image.fromarray(mask_pred * 255)
        image_pred = Image.new("RGB", (W, H))
        image_pred.paste(mask_pred)
        draw_pred = ImageDraw.Draw(image_pred)
        for box in box_pred:
            draw_pred.rectangle(list(box), outline="blue", width=2)

        img_source_ori = Image.open(file_name)
        img_source_ori = img_source_ori.resize((W, H))
        filterd_pred_det_box = pred_box_parts[np.where(pred_det_scores > self.box_threshold)].astype(np.int32)
        draw_alldetmask = ImageDraw.Draw(img_source_ori)

        for ind, box in enumerate(filterd_pred_det_box):
            color = colors[ind]
            draw_alldetmask.rectangle(list(box), outline=color, width=2)

        imge_ori = Image.open(file_name)
        imge_ori = imge_ori.resize((W, H))
        draw_alldet = ImageDraw.Draw(imge_ori)
        for ind, box in enumerate(gt_all_box):
            color = colors[ind]
            draw_alldet.rectangle(list(box), outline=color, width=2)

        imshow_image_nums = 5

        if "attn_map_query" in extra_dict:
            attn_map = attn_map[:, 1 : (50 - min) - 1]
            attn_map = attn_map.cpu().detach().numpy()
            attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))
            attn_map = (attn_map * 255).astype(np.uint8)
            attn_map = cv2.resize(attn_map, (W, H))
            attn_map = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
            attn_map = cv2.cvtColor(attn_map, cv2.COLOR_BGR2RGB)
            attn_map = Image.fromarray(attn_map)
            image_attn = Image.new("RGB", (W, H))
            image_attn.paste(attn_map)

            imshow_image_nums += 1

        img_source = Image.open(file_name)
        img_source = img_source.resize((W, H))
        concat_image = Image.new("RGB", (W * imshow_image_nums + (imshow_image_nums - 1) * 10, H), "white")
        concat_image.paste(img_source, (0, 0))
        concat_image.paste(image_gt, (W + 10, 0))
        concat_image.paste(image_pred, (2 * W + 20, 0))
        concat_image.paste(img_source_ori, (3 * W + 30, 0))
        concat_image.paste(imge_ori, (4 * W + 40, 0))
        if "attn_map_query" in extra_dict:
            concat_image.paste(image_attn, (5 * W + 50, 0))

        save_filename_src = save_filename + "-{}-src.jpg".format(expression)
        concat_image.save(save_filename_src)
