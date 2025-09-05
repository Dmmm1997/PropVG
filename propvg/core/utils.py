import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import pycocotools.mask as maskUtils
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from propvg.utils.visualizer import GenericMask, VisImage
import cv2
import mmcv
import numpy
import torch
from torchvision.ops.boxes import box_area
import random


EPS = 1e-2


def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)


def imshow_expr_bbox(filename, pred_bbox, outfile, gt_bbox=None, pred_bbox_color="r", gt_bbox_color="b", thickness=2):
    # plt.clf()
    # _, axe = plt.subplots()

    # pred_bbox_color = color_val_matplotlib(pred_bbox_color)
    # gt_bbox_color = color_val_matplotlib(gt_bbox_color)

    # img = mmcv.imread(filename, channel_order="rgb").astype(numpy.uint8)
    # img = numpy.ascontiguousarray(img)
    # if pred_bbox is not None and pred_bbox.shape[0]!=0:
    #     if len(pred_bbox.shape)==2:
    #         pred_bboxes = pred_bbox
    #     else:
    #         pred_bboxes = pred_bbox.unsqueeze(0)
    #     for pred_bbox in pred_bboxes:
    #         pred_bbox_int = pred_bbox.long().cpu()
    #         pred_bbox_poly = [[pred_bbox_int[0], pred_bbox_int[1]], [pred_bbox_int[2], pred_bbox_int[1]],
    #                         [pred_bbox_int[2], pred_bbox_int[3]], [pred_bbox_int[0], pred_bbox_int[3]]]
    #         pred_bbox_poly = numpy.array(pred_bbox_poly).reshape((4, 2))
    #         pred_polygon = Polygon(pred_bbox_poly)
    #         pred_patch = PatchCollection([pred_polygon], facecolor='none', edgecolors=[
    #                                     pred_bbox_color], linewidths=thickness)

    #         axe.add_collection(pred_patch)

    # if gt_bbox is not None:
    #     if len(gt_bbox.shape)==2:
    #         gt_bboxes = gt_bbox
    #     else:
    #         gt_bboxes = gt_bbox.unsqueeze(0)
    #     for gt_bbox in gt_bboxes:
    #         gt_bbox_int = gt_bbox.long().cpu()
    #         gt_bbox_poly = [[gt_bbox_int[0], gt_bbox_int[1]], [gt_bbox_int[0], gt_bbox_int[3]],
    #                         [gt_bbox_int[2], gt_bbox_int[3]], [gt_bbox_int[2], gt_bbox_int[1]]]
    #         gt_bbox_poly = numpy.array(gt_bbox_poly).reshape((4, 2))
    #         gt_polygon = Polygon(gt_bbox_poly)
    #         gt_patch = PatchCollection(
    #             [gt_polygon], facecolor='none', edgecolors=[gt_bbox_color], linewidths=thickness)
    #         axe.add_collection(gt_patch)

    # axe.axis('off')
    # axe.imshow(img)
    # plt.savefig(outfile)

    # plt.close()

    img = cv2.imread(filename)
    if pred_bbox is not None and pred_bbox.shape[0] != 0:
        if len(pred_bbox.shape) == 2:
            pred_bboxes = pred_bbox
        else:
            pred_bboxes = pred_bbox.unsqueeze(0)
        for pred_bbox in pred_bboxes:
            pred_bbox_int = pred_bbox.long().cpu().detach().numpy()
            cv2.rectangle(
                img,
                (pred_bbox_int[0], pred_bbox_int[1]),
                (pred_bbox_int[2], pred_bbox_int[3]),
                color=[255, 0, 0],
                thickness=2,
            )

    if gt_bbox is not None:
        if len(gt_bbox.shape) == 2:
            gt_bboxes = gt_bbox
        else:
            gt_bboxes = gt_bbox.unsqueeze(0)
        for gt_bbox in gt_bboxes:
            gt_bbox_int = gt_bbox.long().cpu().detach().numpy()
            cv2.rectangle(
                img, (gt_bbox_int[0], gt_bbox_int[1]), (gt_bbox_int[2], gt_bbox_int[3]), color=[0, 0, 255], thickness=2
            )

    cv2.imwrite(outfile, img)


def is_badcase_boxsegioulowerthanthr(box, seg, thr=0.5):
    seg = maskUtils.decode(seg)
    seg = torch.tensor(seg)
    box = box.int()
    H, W = seg.shape
    box[0] = torch.clamp(box[0], min=0, max=W - 1)  # x1
    box[1] = torch.clamp(box[1], min=0, max=H - 1)  # y1
    box[2] = torch.clamp(box[2], min=0, max=W - 1)  # x2
    box[3] = torch.clamp(box[3], min=0, max=H - 1)  # y2
    iou = compute_segboxiou(seg, box, threshold=0.5)
    return iou < thr


def imshow_expr_mask(filename, pred_mask, outfile, gt_mask=None, overlay=True):
    if not overlay:
        plt.clf()
        plt.axis("off")
        pred_mask = maskUtils.decode(pred_mask).astype(bool)
        plt.imshow(pred_mask, "gray")
        plt.savefig(outfile.replace(".jpg", "_pred.jpg"))
        if gt_mask is not None:
            plt.clf()
            plt.axis("off")
            gt_mask = maskUtils.decode(gt_mask).astype(bool)
            assert gt_mask.shape == pred_mask.shape
            plt.imshow(gt_mask, "gray")
            plt.savefig(outfile.replace(".jpg", "_gt.jpg"))
        plt.close()
    else:
        img = cv2.imread(filename)[:, :, ::-1]
        height, width = img.shape[:2]
        img = numpy.ascontiguousarray(img).clip(0, 255).astype(numpy.uint8)
        output_pred = VisImage(img, scale=1.0)
        pred_mask = maskUtils.decode(pred_mask)
        assert pred_mask.shape[0] == height and pred_mask.shape[1] == width
        pred_mask = GenericMask(pred_mask, height, width)
        for segment in pred_mask.polygons:
            polygon = mpl.patches.Polygon(
                segment.reshape(-1, 2),
                fill=True,
                facecolor=mplc.to_rgb([0.439, 0.188, 0.627]) + (0.65,),
                edgecolor=mplc.to_rgb([0.0, 0.0, 0.0]) + (1,),
                linewidth=2,
            )
            output_pred.ax.add_patch(polygon)
        cv2.imwrite(outfile.replace(".jpg", "_pred.jpg"), output_pred.get_image()[:, :, ::-1])
        if gt_mask is not None:
            output_gt = VisImage(img, scale=1.0)
            gt_mask = maskUtils.decode(gt_mask)
            assert gt_mask.shape[0] == height and gt_mask.shape[1] == width
            gt_mask = GenericMask(gt_mask, height, width)
            for segment in gt_mask.polygons:
                polygon = mpl.patches.Polygon(
                    segment.reshape(-1, 2),
                    fill=True,
                    facecolor=mplc.to_rgb([0.439, 0.188, 0.627]) + (0.65,),
                    edgecolor=mplc.to_rgb([0.0, 0.0, 0.0]) + (1,),
                    linewidth=2,
                )
                output_gt.ax.add_patch(polygon)
            cv2.imwrite(outfile.replace(".jpg", "_gt.jpg"), output_gt.get_image()[:, :, ::-1])


import matplotlib.patches as mpl_patches


def imshow_box_mask(filename, pred_bbox, pred_masks, outfile, gt=False, gres=True):
    # seg
    if gt:
        facecolor = mplc.to_rgb([1.000, 0.627, 0.478]) + (0.65,)  # Light coral with 65% opacity
        edgecolor = mplc.to_rgb([0.545, 0.000, 0.000]) + (1.0,)
        edgecolor_box = "r"
    else:
        facecolor = mplc.to_rgb([0.439, 0.188, 0.627]) + (0.65,)
        edgecolor = mplc.to_rgb([0.0, 0.0, 0.0]) + (1,)
        edgecolor_box = "b"
    img = cv2.imread(filename)[:, :, ::-1]
    height, width = img.shape[:2]
    img = numpy.ascontiguousarray(img).clip(0, 255).astype(numpy.uint8)
    output_pred = VisImage(img, scale=1.0)
    if "pred_masks" in pred_masks:  # judge if it is gres
        pred_nt = pred_masks["pred_nt"].argmax(dim=0).bool()
        size = pred_masks["pred_masks"]["size"]
        if pred_nt:
            pred_mask = numpy.zeros(size, dtype=numpy.uint8)
        else:
            pred_mask = pred_masks["pred_masks"]
            pred_mask = maskUtils.decode(pred_mask)
    else:
        pred_mask = maskUtils.decode(pred_masks)
    assert pred_mask.shape[0] == height and pred_mask.shape[1] == width
    pred_mask = GenericMask(pred_mask, height, width)
    for segment in pred_mask.polygons:
        polygon = mpl.patches.Polygon(
            segment.reshape(-1, 2),
            fill=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2,
        )
        output_pred.ax.add_patch(polygon)

    # Draw bounding boxes if available
    if pred_bbox is not None and pred_bbox.shape[0] != 0:
        if len(pred_bbox.shape) == 2:
            pred_bboxes = pred_bbox
        else:
            pred_bboxes = pred_bbox.unsqueeze(0)
        for pred_bbox in pred_bboxes:
            pred_bbox_int = pred_bbox.long().cpu().detach().numpy()
            rect = mpl_patches.Rectangle(
                (pred_bbox_int[0], pred_bbox_int[1]),  # (x1, y1)
                pred_bbox_int[2] - pred_bbox_int[0],  # width
                pred_bbox_int[3] - pred_bbox_int[1],  # height
                linewidth=2,
                edgecolor=edgecolor_box,
                facecolor="none",
            )
            output_pred.ax.add_patch(rect)

    cv2.imwrite(outfile, output_pred.get_image()[:, :, ::-1])


def imshow_attmap(filename, attmap, outfile, imgsave_file=None):

    img = cv2.imread(filename)
    if imgsave_file is not None:
        cv2.imwrite(imgsave_file, img)
    if attmap is not None:
        attmap_resized = cv2.resize(attmap, (img.shape[1], img.shape[0]))
        attmap_normalized = cv2.normalize(attmap_resized, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        attmap_colormap = cv2.applyColorMap(attmap_normalized, cv2.COLORMAP_JET)
        # 将原图和热力图进行融合，设置 alpha 值控制透明度
        alpha = 0.6  # 透明度
        fused_image = cv2.addWeighted(img, 1 - alpha, attmap_colormap, alpha, 0)

    cv2.imwrite(outfile, fused_image)


def imshow_query(filename, det_bbox, outfile, pred_scores=None, det_scores=None):
    # seg
    img = cv2.imread(filename)[:, :, ::-1]
    height, width = img.shape[:2]
    black_img = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    output_pred = VisImage(black_img, scale=1.0)

    # Draw bounding boxes if available
    if det_bbox is not None and det_bbox.shape[0] != 0:
        if len(det_bbox.shape) == 2:
            pred_bboxes = det_bbox
        else:
            pred_bboxes = det_bbox.unsqueeze(0)
        for ind, pred_bbox in enumerate(pred_bboxes):
            pred_bbox_int = pred_bbox.long().cpu().detach().numpy()

            if pred_scores > 0.7 and det_scores > 0.5:
                rect = mpl_patches.Rectangle(
                    (pred_bbox_int[0], pred_bbox_int[1]),  # (x1, y1)
                    pred_bbox_int[2] - pred_bbox_int[0],  # width
                    pred_bbox_int[3] - pred_bbox_int[1],  # height
                    linewidth=4,
                    edgecolor="r",
                    facecolor="none",
                )
                output_pred.ax.add_patch(rect)
                det_score = det_scores
                ref_score = pred_scores
                output_pred.ax.text(
                    pred_bbox_int[0] - 4,
                    pred_bbox_int[1] + 8,
                    "d:{:.2f}|r:{:.2f}".format(det_score, ref_score),
                    fontsize=22,
                    color="b",
                    bbox=dict(facecolor="white", alpha=0.75),
                )
            else:
                rect = mpl_patches.Rectangle(
                    (pred_bbox_int[0], pred_bbox_int[1]),  # (x1, y1)
                    pred_bbox_int[2] - pred_bbox_int[0],  # width
                    pred_bbox_int[3] - pred_bbox_int[1],  # height
                    linewidth=2,
                    edgecolor="b",
                    facecolor="none",
                )
                output_pred.ax.add_patch(rect)
    cv2.imwrite(outfile, output_pred.get_image()[:, :, ::-1])


# import torch.nn.functional as F
def imshow_box_query(filename, det_bbox, outfile, pred_scores=None):
    img = cv2.imread(filename)[:, :, ::-1]
    height, width = img.shape[:2]
    img = numpy.ascontiguousarray(img).clip(0, 255).astype(numpy.uint8)
    output_pred = VisImage(img, scale=1.0)

    # Draw bounding boxes if available
    if det_bbox is not None and det_bbox.shape[0] != 0:
        if len(det_bbox.shape) == 2:
            pred_bboxes = det_bbox
        else:
            pred_bboxes = det_bbox.unsqueeze(0)
        for ind, pred_bbox in enumerate(pred_bboxes):
            pred_bbox_int = pred_bbox.long().cpu().detach().numpy()

            rect = mpl_patches.Rectangle(
                (pred_bbox_int[0], pred_bbox_int[1]),  # (x1, y1)
                pred_bbox_int[2] - pred_bbox_int[0],  # width
                pred_bbox_int[3] - pred_bbox_int[1],  # height
                linewidth=4,
                edgecolor="r",
                facecolor="none",
            )
            output_pred.ax.add_patch(rect)
            ref_score = pred_scores
            output_pred.ax.text(
                pred_bbox_int[0] - 4,
                pred_bbox_int[1] + 8,
                "score:{:.2f}".format(ref_score),
                fontsize=22,
                color="b",
                bbox=dict(facecolor="white", alpha=0.75),
            )
            output_pred.ax.add_patch(rect)
    cv2.imwrite(outfile, output_pred.get_image()[:, :, ::-1])


def imshow_gres(
    filename,
    refer_bbox,
    det_bbox,
    outfile,
    pred_masks=None,
    pred_scores=None,
    det_scores=None,
    imgsave_file=None,
    gt=False,
):
    # seg
    if gt:
        facecolor = mplc.to_rgb([1.000, 0.627, 0.478]) + (0.65,)  # Light coral with 65% opacity
        edgecolor = mplc.to_rgb([0.545, 0.000, 0.000]) + (1.0,)
        edgecolor_box = "r"
    else:
        facecolor = mplc.to_rgb([0.439, 0.188, 0.627]) + (0.65,)
        edgecolor = mplc.to_rgb([0.0, 0.0, 0.0]) + (1,)
        edgecolor_box = "b"
    img = cv2.imread(filename)[:, :, ::-1]
    height, width = img.shape[:2]
    img = numpy.ascontiguousarray(img).clip(0, 255).astype(numpy.uint8)
    output_pred = VisImage(img, scale=1.0)
    if imgsave_file is not None:
        cv2.imwrite(imgsave_file, output_pred.get_image()[:, :, ::-1])
    if pred_masks is not None:
        if "pred_masks" in pred_masks:  # judge if it is gres
            # pred_nt = pred_masks["pred_nt"].argmax(dim=0).bool()
            # size = pred_masks["pred_masks"]["size"]
            # if pred_nt:
            #     pred_mask = numpy.zeros(size, dtype=numpy.uint8)
            # else:
            pred_mask = pred_masks["pred_masks"]
            pred_mask = maskUtils.decode(pred_mask)
        else:
            pred_mask = maskUtils.decode(pred_masks)
        assert pred_mask.shape[0] == height and pred_mask.shape[1] == width
        pred_mask = GenericMask(pred_mask, height, width)
        for segment in pred_mask.polygons:
            polygon = mpl.patches.Polygon(
                segment.reshape(-1, 2),
                fill=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=2,
            )
            output_pred.ax.add_patch(polygon)

    # Draw bounding boxes if available
    if det_bbox is not None and det_bbox.shape[0] != 0:
        if len(det_bbox.shape) == 2:
            pred_bboxes = det_bbox
        else:
            pred_bboxes = det_bbox.unsqueeze(0)
        for ind, pred_bbox in enumerate(pred_bboxes):
            pred_bbox_int = pred_bbox.long().cpu().detach().numpy()
            rect = mpl_patches.Rectangle(
                (pred_bbox_int[0], pred_bbox_int[1]),  # (x1, y1)
                pred_bbox_int[2] - pred_bbox_int[0],  # width
                pred_bbox_int[3] - pred_bbox_int[1],  # height
                linewidth=4,
                edgecolor="b",
                facecolor="none",
            )
            output_pred.ax.add_patch(rect)
            if det_scores is not None:
                det_score = det_scores[ind]
                output_pred.ax.text(
                    pred_bbox_int[0] - 4,
                    pred_bbox_int[1] + 8,
                    "d:{:.2f}".format(det_score),
                    fontsize=22,
                    color="b",
                    bbox=dict(facecolor="white", alpha=0.75),
                )

    # Draw bounding boxes if available
    if refer_bbox is not None and refer_bbox.shape[0] != 0:
        if len(refer_bbox.shape) == 2:
            pred_bboxes = refer_bbox
        else:
            pred_bboxes = refer_bbox.unsqueeze(0)
        for ind, pred_bbox in enumerate(pred_bboxes):
            pred_bbox_int = pred_bbox.long().cpu().detach().numpy()
            rect = mpl_patches.Rectangle(
                (pred_bbox_int[0], pred_bbox_int[1]),  # (x1, y1)
                pred_bbox_int[2] - pred_bbox_int[0],  # width
                pred_bbox_int[3] - pred_bbox_int[1],  # height
                linewidth=4,
                edgecolor="r",
                facecolor="none",
            )
            output_pred.ax.add_patch(rect)
            if pred_scores is not None:
                # pred_score = pred_scores[ind].cpu().detach().numpy()
                pred_score = pred_scores[ind]
                output_pred.ax.text(
                    pred_bbox_int[0] - 4,
                    pred_bbox_int[1] + 8,
                    "r:{:.2f}".format(pred_score),
                    fontsize=22,
                    color="r",
                    bbox=dict(facecolor="white", alpha=0.75),
                )

    cv2.imwrite(outfile, output_pred.get_image()[:, :, ::-1])


def imshow_box_mask_parts(filename, pred_bbox, pred_masks, outfile, gt=False, gres=True, empty=False):
    def generate_random_color():
        # 生成随机的RGB颜色
        rgb = [random.random() for _ in range(3)]
        # 生成随机的alpha值
        # alpha_face = random.uniform(0.5, 1.0)
        # alpha_edge = random.uniform(0.5, 1.0)

        # 生成facecolor和edgecolor
        facecolor = mplc.to_rgb(rgb) + (0.65,)
        edgecolor_box = mplc.to_rgb(rgb) + (1.0,)
        return facecolor, edgecolor_box

    # seg
    img = cv2.imread(filename)[:, :, ::-1]
    height, width = img.shape[:2]
    img = numpy.ascontiguousarray(img).clip(0, 255).astype(numpy.uint8)
    output_pred = VisImage(img, scale=1.0)
    color_list = []
    if not empty:
        for pred_mask in pred_masks:
            # pred_mask = maskUtils.decode(pred_mask)
            assert pred_mask.shape[0] == height and pred_mask.shape[1] == width
            pred_mask = GenericMask(pred_mask, height, width)
            facecolor, edgecolor_box = generate_random_color()
            color_list.append(edgecolor_box)
            for segment in pred_mask.polygons:
                polygon = mpl.patches.Polygon(
                    segment.reshape(-1, 2),
                    fill=True,
                    facecolor=facecolor,
                    edgecolor="red",
                    linewidth=2,
                )
                output_pred.ax.add_patch(polygon)

    # Draw bounding boxes if available
    if pred_bbox is not None and pred_bbox.shape[0] != 0:
        if len(pred_bbox.shape) == 2:
            pred_bboxes = pred_bbox
        else:
            pred_bboxes = pred_bbox.unsqueeze(0)
        for ind, pred_bbox in enumerate(pred_bboxes):
            pred_bbox_int = pred_bbox.long().cpu().detach().numpy()
            rect = mpl_patches.Rectangle(
                (pred_bbox_int[0], pred_bbox_int[1]),  # (x1, y1)
                pred_bbox_int[2] - pred_bbox_int[0],  # width
                pred_bbox_int[3] - pred_bbox_int[1],  # height
                linewidth=2,
                edgecolor=color_list[ind],
                facecolor="none",
            )
            output_pred.ax.add_patch(rect)

    cv2.imwrite(outfile, output_pred.get_image()[:, :, ::-1])


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
