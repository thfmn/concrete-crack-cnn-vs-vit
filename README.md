# Concrete Crack Detection: CNN vs Vision Transformer Benchmark

Benchmarking modern CNN architectures against Vision Transformers on concrete crack detection (classification) and segmentation tasks.

## Planned Models

| Model | Family | Params | timm key |
|-------|--------|--------|----------|
| ResNet-50 | CNN | ~25M | `resnet50` |
| EfficientNetV2-S | CNN | ~22M | `tf_efficientnetv2_s` |
| ConvNeXt-Tiny | CNN | ~29M | `convnext_tiny` |
| ViT-B/16 | ViT | ~86M | `vit_base_patch16_224` |
| Swin-T | ViT | ~28M | `swin_tiny_patch4_window7_224` |
| DeiT-Small | ViT | ~22M | `deit_small_patch16_224` |

## Stack

PyTorch · Lightning · timm · segmentation-models-pytorch · albumentations · Hydra · MLflow

## Tasks

- **Classification** — Crack vs no-crack on SDNET2018 (~56k images)
- **Segmentation** — Pixel-level crack masks on CrackSeg9k (~9k images)

## Setup

```bash
uv sync
uv run pytest tests/ -x -q
```

## License

MIT © 2026 Tobias Hoffmann
