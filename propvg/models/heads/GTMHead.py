import copy
import torch
from torch import nn
from torch.nn import functional as F
import torch
from propvg.core.criterion.criterion import SetCriterion
from propvg.core.matcher import HungarianMatcher
from propvg.layers.box_ops import box_xyxy_to_cxcywh
from propvg.layers.mlp import MLP
from propvg.layers.position_embedding import (
    PositionEmbeddingSine,
    PositionEmbeddingSine1D,
)
from propvg.models import HEADS
import pycocotools.mask as maskUtils
import numpy as np

from propvg.models.heads.tgqs_kd_detr_head.transformer import DetrTransformerDecoder
from propvg.models.heads.transformers.deformable_transformer import (
    DeformableDetrTransformer,
    DeformableDetrTransformerDecoder,
    inverse_sigmoid,
)
from .modules import SegBranch
from .unet_head import SimpleFPN, UnetDecoder
from propvg.models.losses.segloss import seg_loss


class GlobalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, rpe_hidden_dim=512):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.score_mlp = self.build_cpb_mlp(1, rpe_hidden_dim, num_heads)

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim, bias=False)
        )
        return cpb_mlp

    def forward(
        self,
        query,
        k_input_flatten,
        v_input_flatten,
        input_padding_mask=None,
        attn_mask=None,  # B,1,N
    ):

        attn_mask = self.score_mlp(attn_mask.unsqueeze(-1)).permute(0, 3, 1, 2)  # (B,1,N,nhead)

        B_, N, C = k_input_flatten.shape
        k = self.k(k_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            assert attn.shape == attn_mask.shape
            attn += attn_mask

        if input_padding_mask is not None:
            # attn += input_padding_mask[:, None, None] * -100
            input_mask = input_padding_mask.reshape(input_padding_mask.size(0), 1, 1, -1).repeat(
                1, self.num_heads, 1, 1
            )
            attn += input_mask * -100
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class GlobalDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
        norm_type="post_norm",
    ):
        super().__init__()

        self.norm_type = norm_type

        # global cross attention
        self.cross_attn = GlobalCrossAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(
        self, tgt, query_pos, src, src_pos_embed, src_padding_mask=None, self_attn_mask=None, cross_attn_mask=None
    ):
        # self attention
        tgt2 = self.norm2(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt2.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[
            0
        ].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)

        # global cross attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt2, query_pos),
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_padding_mask,
            cross_attn_mask,
        )
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

    def forward_post(
        self, tgt, query_pos, src, src_pos_embed, src_padding_mask=None, self_attn_mask=None, cross_attn_mask=None
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[
            0
        ].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_padding_mask,
            cross_attn_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward(
        self, tgt, query_pos, src, src_pos_embed, src_padding_mask=None, self_attn_mask=None, cross_attn_mask=None
    ):
        if self.norm_type == "pre_norm":
            return self.forward_pre(
                tgt, query_pos, src, src_pos_embed, src_padding_mask, self_attn_mask, cross_attn_mask
            )
        if self.norm_type == "post_norm":
            return self.forward_post(
                tgt, query_pos, src, src_pos_embed, src_padding_mask, self_attn_mask, cross_attn_mask
            )


class DETRLoss(nn.Module):
    def __init__(
        self,
        criterion={"loss_class": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
        matcher={"cost_class": 1.0, "cost_bbox": 5.0, "cost_giou": 2.0},
        aux_loss=True,
        num_classes=1,
    ):
        super(DETRLoss, self).__init__()
        self.aux_loss = aux_loss
        self.matcher = HungarianMatcher(
            cost_class=matcher["cost_class"],
            cost_bbox=matcher["cost_bbox"],
            cost_giou=matcher["cost_giou"],
            cost_class_type="ce_cost",
        )
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=self.matcher,
            weight_dict={
                "loss_class": criterion["loss_class"],
                "loss_bbox": criterion["loss_bbox"],
                "loss_giou": criterion["loss_giou"],
            },
            loss_class_type="ce_loss",
            eos_coef=0.1,
        )

    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def forward(self, output_class, output_coord, targets):
        output = {
            "pred_logits": output_class[-1],
            "pred_boxes": output_coord[-1],
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(output_class, output_coord)
        loss_dict, indices = self.criterion(output, targets, return_indices=True)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict, indices

    def forward_withmatching(self, output_class, output_coord, targets, matching_list):
        output = {
            "pred_logits": output_class[-1],
            "pred_boxes": output_coord[-1],
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(output_class, output_coord)
        loss_dict, indices = self.criterion(output, targets, matching_list, return_indices=True)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict, indices


class CRS(nn.Module):
    def __init__(self, hidden_channels, temperature_learnable=False):
        super(CRS, self).__init__()
        if temperature_learnable:
            self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        else:
            self.temperature = 0.07

        self.word_attn = DetrTransformerDecoder(
            embed_dim=hidden_channels,
            num_heads=8,
            attn_dropout=0.1,
            feedforward_dim=512,
            ffn_dropout=0.1,
            num_layers=3,
            return_intermediate=False,
            post_norm=True,
            batch_first=True,
        )
        self.refer_linear = nn.Linear(hidden_channels, 1, bias=False)
        self.position_embedding_1d = PositionEmbeddingSine1D(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )

    def text_pooler(self, lan_feat, lan_mask):
        lan_feat_pooler = torch.cat(
            list(
                map(
                    lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0],
                    lan_feat,
                    ~lan_mask,
                )
            )
        )
        return lan_feat_pooler

    def forward(self, query_feat, lan_feat, lan_mask):
        text_pos_embed = self.position_embedding_1d(lan_feat).unsqueeze(0).repeat(lan_feat.shape[0], 1, 1).cuda()
        # Word level Attention
        query_embed, proposal_word_attnmap = self.word_attn(
            query=query_feat,
            key=lan_feat,
            value=lan_feat,
            key_pos=text_pos_embed,
            key_padding_mask=lan_mask.bool(),
        )

        # Sent level Contrastive
        lan_feat_maxpooling = self.text_pooler(lan_feat, lan_mask).unsqueeze(1)  # B,1,256
        similarity_sent = (
            F.cosine_similarity(query_embed[-1].unsqueeze(2), lan_feat_maxpooling.unsqueeze(1), dim=-1)
            / self.temperature
        )
        query_sent_cons_score, _ = similarity_sent.max(dim=-1, keepdim=True)
        # Word level Contrastive
        similarity_word = (
            F.cosine_similarity(query_embed[-1].unsqueeze(2), lan_feat.unsqueeze(1), dim=-1) / self.temperature
        )
        query_word_score, query_word_index = similarity_word.max(dim=-1, keepdim=True)
        weight = (self.refer_linear(lan_feat_maxpooling).reshape(lan_feat_maxpooling.size(0), 1, 1)).sigmoid()
        refer_score = (query_sent_cons_score * weight + query_word_score * (1 - weight)).sigmoid()

        return refer_score, proposal_word_attnmap


class MTDv2(nn.Module):
    def __init__(self, hidden_channels, K):
        super(MTDv2, self).__init__()
        self.DETAttn = GlobalDecoderLayer(
            d_model=hidden_channels, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8
        )
        self.SegAttn = GlobalDecoderLayer(
            d_model=hidden_channels, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8
        )
        self.distinguish_linear = MLP(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            output_dim=1,
            num_layers=3,
        )
        self.query_nt = nn.Embedding(1, hidden_channels)
        self.global_max_pool = nn.AdaptiveMaxPool2d((3, 3))
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.k = K

    def x_mask_pos_enc(self, x, img_shape):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_shape
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape
            x_mask[img_id, :img_h, :img_w] = 0
        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)
        x_pos_embeds = self.position_embedding(x_mask)
        return x_mask, x_pos_embeds

    def forward(self, query_feat, unet_feat, refer_score, mask_score):

        query_nt = self.query_nt.weight.unsqueeze(0).repeat(query_feat.shape[0], 1, 1)
        batch_size, hidden_size = unet_feat.size(0), unet_feat.size(1)

        nt_embed = self.DETAttn(
            tgt=query_nt,
            query_pos=None,
            src=query_feat,
            src_pos_embed=None,
            src_padding_mask=None,
            self_attn_mask=None,
            cross_attn_mask=refer_score.transpose(1, 2),
        )
        _, pos_embed = self.x_mask_pos_enc(mask_score, mask_score.shape[-2:])
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
        nt_feat = self.SegAttn(
            tgt=nt_embed,
            query_pos=None,
            src=unet_feat.reshape(batch_size, -1, hidden_size),
            src_pos_embed=pos_embed,
            src_padding_mask=None,
            self_attn_mask=None,
            cross_attn_mask=mask_score.reshape(mask_score.shape[0], 1, -1),
        )

        nt_pred = self.distinguish_linear(nt_feat).view(batch_size, 1).sigmoid()
        nt_pred_ref = refer_score.squeeze(-1).max(dim=-1, keepdim=True)[0].view(batch_size, 1)
        if self.training:
            nt_pred_mask, _ = torch.topk(mask_score.view(batch_size, -1), k=10, dim=-1)
        else:
            nt_pred_mask, _ = torch.topk(mask_score.view(batch_size, -1), k=self.k, dim=-1)
        nt_pred_mask = nt_pred_mask.mean(dim=-1).view(batch_size, 1)
        nt_pred = nt_pred * nt_pred_ref * nt_pred_mask

        return nt_pred


@HEADS.register_module()
class GTMHead(nn.Module):
    def __init__(
        self,
        input_channels=768,
        hidden_channels=256,
        loss_weight={"mask": 1.0, "bbox": 0.025, "cons": 0.0},
        num_queries=20,
        detr_loss={},
        MTD={},
    ):
        super(GTMHead, self).__init__()
        self.seg_branch = SegBranch(hidden_channels, upsample_rate=1)
        if "aux" in loss_weight["mask"]:
            self.seg_branch_aux_list = nn.ModuleList(
                [SegBranch(hidden_channels // 4 * (2**i), upsample_rate=1) for i in range(4)]
            )
        self.nt_embed = MLP(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            output_dim=1,
            num_layers=3,
        )
        self.class_embed = nn.Linear(hidden_channels, 1 + 1)
        self.bbox_embed = MLP(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            output_dim=4,
            num_layers=3,
        )
        self.refer_embed = MLP(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            output_dim=1,
            num_layers=3,
        )

        self.loss_weight = loss_weight
        self.lan_embedding = nn.Linear(input_channels, hidden_channels, bias=False)
        hidden_channels_ = hidden_channels * 2
        self.img_embedding = nn.Conv2d(input_channels, hidden_channels_, kernel_size=1, bias=False)
        self.neck = SimpleFPN(
            backbone_channel=hidden_channels_,
            in_channels=[
                hidden_channels_ // 4,
                hidden_channels_ // 2,
                hidden_channels_,
                hidden_channels_,
            ],
            out_channels=[
                hidden_channels_,
                hidden_channels_,
                hidden_channels_,
                hidden_channels_,
            ],
        )
        self.unet_decoder = UnetDecoder(hidden_channels, 1)

        self.detrloss = DETRLoss(
            aux_loss=True,
            criterion=detr_loss["criterion"],
            matcher=detr_loss["matcher"],
            num_classes=1,
        )
        self.referloss = nn.BCEWithLogitsLoss()
        self.decoder_transformer = DeformableDetrTransformer(
            encoder=None,
            decoder=DeformableDetrTransformerDecoder(
                embed_dim=256,
                num_heads=8,
                feedforward_dim=1024,
                attn_dropout=0.1,
                ffn_dropout=0.1,
                num_layers=3,
                return_intermediate=True,
            ),
            only_decoder=True,
        )
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        num_pred = self.decoder_transformer.decoder.num_layers
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

        self.query_pos_embedding = nn.Linear(2, hidden_channels)
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_channels // 2,
            temperature=10000,
            normalize=True,
        )
        self.refer_adapter = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.query_embed = nn.Embedding(num_queries, hidden_channels)
        self.query_pos = nn.Embedding(num_queries, hidden_channels)
        self.refer_loss = nn.CrossEntropyLoss()
        self.mrsm = CRS(hidden_channels)
        self.dist_head = MTDv2(hidden_channels, MTD["K"])
        self.num_queries = num_queries
        self.refer_class_embed = nn.Linear(hidden_channels, 1 + 1)
        self.refer_bbox_embed = MLP(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            output_dim=4,
            num_layers=3,
        )
        self.query_linear_scoring = nn.Linear(hidden_channels, 1)

    def prepare_targets(self, targets, img_metas):
        new_targets = []
        is_empty = []
        refer_indices = [meta["refer_target_index"] for meta in img_metas]
        for ind, target_bbox, img_meta in zip(refer_indices, targets, img_metas):
            h, w = img_meta["img_shape"][:2]
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=target_bbox.device)
            if len(target_bbox) == 0:
                target_bbox = torch.zeros((0, 4), device=target_bbox.device)
            else:
                target_bbox = target_bbox.reshape(-1, 4)
            if len(ind) == 0:
                is_empty.append(0)
            else:
                is_empty.append(1)
            gt_classes = torch.zeros(target_bbox.shape[0], device=target_bbox.device).long()
            gt_boxes = target_bbox.float() / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        is_empty = torch.tensor(is_empty, device=target_bbox.device).long()
        return new_targets, is_empty

    def prepare_referring_targets(self, valid_indices, refer_index, query_num):
        B = len(valid_indices)
        result = torch.zeros((B, query_num), dtype=torch.float32).cuda()
        for i, (inds, refer_ind) in enumerate(zip(valid_indices, refer_index)):
            query_ind, target_ind = inds
            # find the relative refered query index
            shot_inds = [i for i, tar_ind in enumerate(target_ind) if tar_ind in refer_ind]
            # get the real refered query index
            shot_query_inds = query_ind[shot_inds]
            result[i, shot_query_inds] = 1
        return result

    def x_mask_pos_enc(self, x, img_shape):
        batch_size = x.size(0)
        input_img_h, input_img_w = img_shape
        x_mask = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape
            x_mask[img_id, :img_h, :img_w] = 0

        x_mask = F.interpolate(x_mask.unsqueeze(1), size=x.size()[-2:]).to(torch.bool).squeeze(1)
        x_pos_embeds = self.position_embedding(x_mask)
        return x_mask, x_pos_embeds

    def forward_train(self, x, targets, cls_feat=None, lan_feat=None, lan_mask=None, img=None):
        device = x.device
        extra_dict = {}
        target_mask = torch.from_numpy(
            np.concatenate([maskUtils.decode(target)[None] for target in targets["mask"]])
        ).to(device)
        img_metas = targets["img_metas"]
        refer_indices = [meta["refer_target_index"] for meta in img_metas]

        # all feats embedding to hidden_channels
        img_feat = self.img_embedding(x)
        lan_feat = self.lan_embedding(lan_feat)

        # neck
        x_c1, x_c2, x_c3, x_c4 = self.neck(img_feat)

        multi_level_feats = [x_c1[:, :256, :, :], x_c2[:, :256, :, :], x_c3[:, :256, :, :], x_c4[:, :256, :, :]]
        seg_feat = [
            x_c4[:, 256:, :, :],
            x_c3[:, 256:, :, :],
            x_c2[:, 256:, :, :],
            x_c1[:, 256:, :, :],
        ]

        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            img_masks, pos_embed = self.x_mask_pos_enc(feat, feat.shape[-2:])
            multi_level_masks.append(img_masks.to(torch.bool))
            multi_level_position_embeddings.append(pos_embed)

        query_embed_input = self.query_embed.weight.unsqueeze(0).repeat(cls_feat.shape[0], 1, 1)
        query_pos_input = self.query_pos.weight.unsqueeze(0).repeat(cls_feat.shape[0], 1, 1)
        # decoder
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            att_map,
        ) = self.decoder_transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embed_input,
            query_pos_input,
        )  # (B, N, C)
        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # segmentaton branch
        pred_mask = self.unet_decoder(*seg_feat)  # (B, C, H, W)
        pred_segment = self.seg_branch(pred_mask)  # (B, 1, H, W)
        pred_seg = F.interpolate(pred_segment, size=img.shape[-2:], mode="bilinear", align_corners=True)

        # refer interaction
        refer_query_embed = self.refer_adapter(inter_states[-1])

        refer_label, proposal_word_attnmap = self.mrsm(refer_query_embed, lan_feat, lan_mask)
        extra_dict["attn_map_query"] = proposal_word_attnmap[0]

        # nt branch
        nt_score = self.dist_head(refer_query_embed, pred_mask, refer_label, pred_segment.sigmoid())  # (B, 1)
        refer_score = refer_label * nt_score.unsqueeze(-1)

        # detection loss
        # stage1 loss
        target_gt, is_refer_empty = self.prepare_targets(targets["bbox"], img_metas)
        # Use all layer outputs to do losses and get all objects
        det_loss_dict, indices_foreground = self.detrloss(outputs_class, outputs_coord, target_gt)  # [1,32,20,2]
        loss_det = sum(det_loss_dict.values()) * self.loss_weight["allbbox"]

        # refer loss
        # prepare the target for referring
        target_gt_refer = self.prepare_referring_targets(indices_foreground[-1], refer_indices, self.num_queries)
        loss_refer = (
            F.binary_cross_entropy(refer_label.reshape(-1), target_gt_refer.reshape(-1)) * self.loss_weight["refer"]
        )

        # seg loss
        loss_mask = seg_loss(pred_seg, target_mask, self.loss_weight["mask"])

        loss_label_nt = (
            F.binary_cross_entropy(nt_score.reshape(-1), is_refer_empty.reshape(-1).float(), reduction="mean")
            * self.loss_weight["mask"]["nt"]
        )

        loss_dict = {
            "loss_mask": loss_mask,
            "loss_det": loss_det,
            "loss_label_nt": loss_label_nt,
            "loss_refer": loss_refer,
        }

        pred_dict = {
            "pred_global_mask": pred_seg.detach(),
            "pred_class_allobj": outputs_class[-1].detach(),
            "pred_bbox_allobj": outputs_coord[-1].detach(),
            "nt_label": nt_score.detach(),
            "refer_label": refer_score.detach(),
            "att_map": att_map[-1].detach(),
        }
        return loss_dict, pred_dict, extra_dict

    def forward_test(self, x, cls_feat=None, lan_feat=None, lan_mask=None, img=None, targets=None):
        extra_dict = {}

        # all feats embedding to hidden_channels
        img_feat = self.img_embedding(x)
        lan_feat = self.lan_embedding(lan_feat)

        # ! neck
        x_c1, x_c2, x_c3, x_c4 = self.neck(img_feat)

        multi_level_feats = [x_c1[:, :256, :, :], x_c2[:, :256, :, :], x_c3[:, :256, :, :], x_c4[:, :256, :, :]]
        seg_feat = [
            x_c4[:, 256:, :, :],
            x_c3[:, 256:, :, :],
            x_c2[:, 256:, :, :],
            x_c1[:, 256:, :, :],
        ]
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            img_masks, pos_embed = self.x_mask_pos_enc(feat, feat.shape[-2:])
            multi_level_masks.append(img_masks.to(torch.bool))
            multi_level_position_embeddings.append(pos_embed)

        query_embed_input = self.query_embed.weight.unsqueeze(0).repeat(cls_feat.shape[0], 1, 1)
        query_pos_input = self.query_pos.weight.unsqueeze(0).repeat(cls_feat.shape[0], 1, 1)
        # ! decoder
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            att_map,
        ) = self.decoder_transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embed_input,
            query_pos_input,
        )  # (B, N, C)
        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # ! segmentaton branch
        pred_mask = self.unet_decoder(*seg_feat)  # (B, C, H, W)
        pred_segment = self.seg_branch(pred_mask)  # (B, 1, H, W)
        pred_seg = F.interpolate(pred_segment, size=img.shape[-2:], mode="bilinear", align_corners=True)

        # refer interaction
        refer_query_embed = self.refer_adapter(inter_states[-1])

        # refer_label - sigmoided
        refer_label, proposal_word_attnmap = self.mrsm(refer_query_embed, lan_feat, lan_mask)
        extra_dict["attn_map_query"] = proposal_word_attnmap[0]

        # nt_label - no sigmoided
        # ! nt branch
        nt_score = self.dist_head(refer_query_embed, pred_mask, refer_label, pred_segment.sigmoid())  # (B, 1)
        refer_score = refer_label * nt_score.unsqueeze(-1)

        pred_dict = {
            "pred_global_mask": pred_seg.detach(),
            "pred_class_allobj": outputs_class[-1].detach(),
            "pred_bbox_allobj": outputs_coord[-1].detach(),
            "nt_label": nt_score.detach(),
            "refer_label": refer_score.detach(),
            "att_map": att_map[-1].detach(),
        }
        return pred_dict, extra_dict
