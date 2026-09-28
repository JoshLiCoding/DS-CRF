"""Save pseudolabels for a subsample of the training set, one PNG per image and variant:
image, GT, hard (argmax of softmax), soft (softmax) and soft (softmax + min-max).

Pseudolabels are produced exactly as in the training loop in main.py, but on un-augmented
inputs (the validation transform: resize + normalize).

    python utils/vis_pseudolabels.py [--config config.yaml]    (run from the repo root)
"""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dino import DinoWSSS
from model.dino_txt_full_img import build_text_embeddings, generate_pseudolabels_batch, get_class_names
from utils.dataset import CustomSegmentationVal, build_dataset
from utils.vis import visualize_soft_probabilities

DINOV3_LOCATION = '/u501/j234li/reg_loss/model/dinov3'
sys.path.append(DINOV3_LOCATION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    cfg = OmegaConf.load(parser.parse_args().config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(cfg.directories.output, 'pseudolabel_vis')
    os.makedirs(output_dir, exist_ok=True)

    dataset = build_dataset(cfg, 'train')
    val_transform_dataset = CustomSegmentationVal(dataset, resize_size=cfg.dataset.resize_size)

    # Text embeddings, as in main.py
    text_model, tokenizer = torch.hub.load(
        DINOV3_LOCATION,
        'dinov3_vitl16_dinotxt_tet1280d20h24l',
        source='local',
        weights=os.path.join(DINOV3_LOCATION, 'weights', 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'),
        backbone_weights=os.path.join(DINOV3_LOCATION, 'weights', 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    )
    text_model = text_model.to(device).eval()
    fg_class_names, bg_class_names = get_class_names(cfg.dataset.dataset_name)
    text_emb_all = build_text_embeddings(text_model, tokenizer, fg_class_names + bg_class_names, device)
    NUM_FG = len(fg_class_names)

    # Pseudolabels only use the frozen backbone + dino.txt head, so the untrained decoder is irrelevant
    model = DinoWSSS(
        backbone_name=cfg.model.backbone_name,
        num_transformer_blocks=cfg.model.num_transformer_blocks,
        num_conv_blocks=cfg.model.num_conv_blocks,
        out_channels=cfg.model.out_channels,
        use_bottleneck=cfg.model.use_bottleneck,
        use_transpose_conv=cfg.model.use_transpose_conv
    ).to(device).eval()

    for idx in range(0, len(dataset), cfg.visualization.train_sample_interval):
        img, gt = dataset[idx]
        image = val_transform_dataset[idx][0].unsqueeze(0).to(device)

        # Image-level tags, as in CustomSegmentationTrain
        present = torch.zeros(cfg.model.num_classes, dtype=torch.bool)
        present[0] = True
        present[torch.from_numpy(dataset.labels(idx))] = True
        present = present.unsqueeze(0).to(device)

        with torch.no_grad():
            model_outputs = model(image)
            size = model_outputs['seg'].shape[-2:]
            args = (model_outputs['dinotxt'], present, text_emb_all, NUM_FG, size, cfg.pseudolabel.temperature)
            softmax_probs = generate_pseudolabels_batch(*args, min_max=False)[0]  # [C, H/4, W/4]
            minmax_probs = generate_pseudolabels_batch(*args, min_max=True)[0]

        hard = softmax_probs.argmax(0).cpu().numpy().astype(np.uint8)

        name = dataset.names[idx]
        h, w = size
        img.resize((w, h), Image.BILINEAR).save(os.path.join(output_dir, f'{name}_image.png'))
        Image.fromarray(dataset.decode_target(np.array(gt.resize((w, h), Image.NEAREST)))).save(os.path.join(output_dir, f'{name}_gt.png'))
        Image.fromarray(dataset.decode_target(hard)).save(os.path.join(output_dir, f'{name}_hard.png'))
        Image.fromarray(visualize_soft_probabilities(softmax_probs, softmax=False)).save(os.path.join(output_dir, f'{name}_softmax.png'))
        Image.fromarray(visualize_soft_probabilities(minmax_probs, softmax=False)).save(os.path.join(output_dir, f'{name}_softmax_minmax.png'))
        print(f"[{idx}] {name} saved")


if __name__ == "__main__":
    main()
