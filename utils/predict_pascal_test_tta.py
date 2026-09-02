"""Predict PASCAL VOC 2012 segmentation *test* masks, with the same flip TTA as evaluate_tta.py.

The test masks are held out, so instead of scoring we write one indexed PNG per image in
the layout the PASCAL evaluation server expects:

    <out>/results/VOC2012/Segmentation/comp6_test_cls/<image_id>.png
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.dino import DinoWSSS
from utils.dataset import CustomSegmentationTest, VOCSegmentationTest, cmap
from utils.evaluate_tta import predict

NUM_CLASSES = 21
PALETTE = cmap().flatten().tolist()
RESULT_DIR = os.path.join('results', 'VOC2012', 'Segmentation', 'comp6_test_cls')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', help='path to the model checkpoint (.pt)')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--out', default='outputs/voc_test', help='directory to write results/ into')
    parser.add_argument('--flip', action='store_true',
                        help='average the segmentation of the image and of its horizontal flip')
    parser.add_argument('--resize', type=int, default=None,
                        help='single input resolution, overriding dataset.resize_size')
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    cfg = OmegaConf.load(args.config)
    resize_size = args.resize or cfg.dataset.resize_size

    test_dataset = CustomSegmentationTest(VOCSegmentationTest(cfg.dataset.root), resize_size=resize_size)

    model = DinoWSSS(
        backbone_name=cfg.model.backbone_name,
        num_transformer_blocks=cfg.model.num_transformer_blocks,
        num_conv_blocks=cfg.model.num_conv_blocks,
        out_channels=NUM_CLASSES,
        use_bottleneck=cfg.model.use_bottleneck,
        use_transpose_conv=cfg.model.use_transpose_conv,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    out_dir = os.path.join(args.out, RESULT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Predicting voc test at {resize_size}px"
          f"{', flip-averaged' if args.flip else ''} ({len(test_dataset)} images) -> {out_dir}")

    with torch.no_grad():
        for image, name, size in tqdm(test_dataset, desc="Predicting"):
            segmentation = predict(model, image.unsqueeze(0).to(device), args.flip)
            segmentation = F.interpolate(segmentation, size=size, mode='bilinear', align_corners=False)

            mask = Image.fromarray(segmentation.argmax(dim=1)[0].byte().cpu().numpy(), mode='P')
            mask.putpalette(PALETTE)
            mask.save(os.path.join(out_dir, name + '.png'))

    print(f"Done. Package for submission with: tar -C {args.out} -czf results.tgz results")


if __name__ == "__main__":
    main()
