_base_ = [
    "../_base_/datasets/segmentation/refcoco-unc.py",
    "../_base_/misc.py",
]

dataset = "RefCOCOUNC"
max_token = 20
img_size = 384
patch_size = 16

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile_TO",
        max_token=max_token,
        with_mask=True,
        with_bbox=True,
        dataset=dataset,
        use_token_type="beit3",
        refer_file="data/seqtr_type/annotations/mixed-seg/coco_all.json",
        object_area_filter=100,
        object_area_rate_filter=[0.05, 0.8],
    ),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=[
            "img",
            "ref_expr_inds",
            "text_attention_mask",
            "gt_mask_rle",
            "gt_bbox",
        ],
        meta_keys=[
            "filename",
            "expression",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "gt_ori_mask",
            "target",
            "empty",
            "refer_target_index",
        ],
    ),
]

val_pipeline = [
    dict(
        type="LoadImageAnnotationsFromFile_TO",
        max_token=max_token,
        with_mask=True,
        with_bbox=True,
        dataset=dataset,
        use_token_type="beit3",
        refer_file="data/seqtr_type/annotations/mixed-seg/coco_all.json",
        object_area_filter=100,
        object_area_rate_filter=[0.05, 0.8],
    ),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(
        type="CollectData",
        keys=["img", "ref_expr_inds", "text_attention_mask", "gt_mask_rle", "gt_bbox"],
        meta_keys=[
            "filename",
            "expression",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "gt_ori_mask",
            "target",
            "empty",
            "refer_target_index",
        ],
    ),
]
test_pipeline = val_pipeline.copy()

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=val_pipeline,
    ),
    testA=dict(
        pipeline=val_pipeline,
    ),
    testB=dict(
        pipeline=val_pipeline,
    ),
)

model = dict(
    type="MIXRefUniModel_OMG",
    vis_enc=dict(
        type="BEIT3",
        img_size=img_size,
        patch_size=patch_size,
        vit_type="base",
        drop_path_rate=0.1,
        vocab_size=64010,
        freeze_layer=-1,
        vision_embed_proj_interpolate=False,
        pretrain="pretrain_weights/beit3_base_patch16_224.zip",
    ),
    lan_enc=None,
    fusion=None,
    head=dict(
        type="REFHead",
        input_channels=768,
        hidden_channels=256,
        num_queries=20,
        detr_loss={
            "criterion": {"loss_class": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            "matcher": {"cost_class": 1.0, "cost_bbox": 5.0, "cost_giou": 2.0},
        },
        loss_weight={"mask": {"dice": 1.0, "bce": 1.0, "nt": 0.2, "neg": 0}, "bbox": 0.1, "allbbox": 0.1, "refer": 1.0},
        MTD={"K": 250},
    ),
    post_params={
        "score_weighted": False,
        "mask_threshold": 0.5,
        "score_threshold": 0.7,
        "with_nms": False,
        "with_mask": True,
    },
    process_visual=True,
    visualize_params={"row_columns": (4, 5)},
)

grad_norm_clip = 0.15
use_fp16 = False
ema = False

lr = 0.0005
optimizer_config = dict(
    type="Adam",
    lr=lr,
    lr_vis_enc=lr / 10.0,
    lr_lan_enc=lr,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0,
    amsgrad=True,
)

scheduler_config = dict(
    type="MultiStepLRWarmUp",
    warmup_epochs=1,
    decay_steps=[21, 27],
    decay_ratio=0.1,
    max_epoch=30,
)

log_interval = 50

start_save_checkpoint = 20
