# Mask Image Watermarking

Official implementation of [Mask Image Watermarking](http://arxiv.org/abs/2504.12739).

## 🔗 Model Weights

We provide pre-trained model weights for inference. You can download them from the following link: **[Download Model Weights](https://huggingface.co/Runyi-Hu/MaskMark)**.

Specifically, the following two variants are available:

- **[D_32bits](https://huggingface.co/Runyi-Hu/MaskMark/resolve/main/D_32bits.pth?download=true), [D_64bits](https://huggingface.co/Runyi-Hu/MaskMark/resolve/main/D_64bits.pth?download=true), [D_128bits](https://huggingface.co/Runyi-Hu/MaskMark/resolve/main/D_128bits.pth?download=true)** – for global watermark embedding with different bits.
- **[ED_32bits](https://huggingface.co/Runyi-Hu/MaskMark/resolve/main/ED_32bits.pth?download=true), [ED_64bits](https://huggingface.co/Runyi-Hu/MaskMark/resolve/main/ED_64bits.pth?download=true), [ED_128bits](https://huggingface.co/Runyi-Hu/MaskMark/resolve/main/ED_128bits.pth?download=true)** – for adaptive local watermark embedding based on the mask with different bits.

After downloading, place the weights into the `checkpoints/` directory.

## 🔍 Inference

### MaskMark-D

```bash
python3 inference.py \
    --device "cuda:0" \
    --model_name "D_32bits" \
    --image_name "00"
```

This command performs the following steps:

1. **Global Watermark Embedding.** A full-image watermark is embedded into the specified image.

2. **Masked Fusion.** Using a predefined mask, only the masked region retains the watermark, while the rest of the image is replaced by the original image. This creates a fused image that contains both watermarked and clean areas.

3. **Watermark Localization & Extraction.** The model then performs watermark localization and extraction on the fused image.

All results will be saved in the `results/D_32bits` directory.

### MaskMark-ED

```bash
python3 inference.py \
    --device "cuda:0" \
    --model_name "ED_32bits" \
    --image_name "00"
```
Unlike **MaskMark-D**, this command enables adaptive **local** watermark embedding during generation. The watermark is primarily embedded within the mask-selected region, while the rest of the image is designed to contain minimal or no watermark signal.

All results will be saved in the `results/ED_32bits` directory.

## 🛠️ Training

### 📦 Data Download

1. Download the [coco 2014 data](https://cocodataset.org/#download).
2. Expected directory structure:
    ```
    data/
    └── coco_data/
        ├── annotations/
        ├── train2014/
        └── val2014/
    ```

### 🚀 Training Scripts

1. **Set dataset path**: Modify the `dataset_path` field in `configs/train/train.yaml` to point to your local dataset directory. For example:
   ```yaml
   dataset_path: data/coco_data
   ```
2. **Select model variant**: You can train different models by changing the model_name in the config file. Supported options: D_32bits, D_64bits, D_128bits, ED_32bits, ED_64bits, ED_128bits.
3. **Start training with the following command (D_32bits)**:
    ```bash
    python3 train.py \
        --model_name "D_32bits" \
        --train_config_path "configs/train/train.yaml" \
        --model_config_path "configs/model/D_32bits.yaml"
    ```
    Logs and checkpoints are saved in the `checkpoints/<model_name>` directory.

### 🎯 Finetuning Scripts

In practical scenarios, you might want to enhance the model's robustness against specific distortions based on your application needs. We provide dedicated finetuning scripts to support this.
Specifically, in Step 3 above, you can simply replace `train.yaml` with `finetune_<distortion_name>.yaml` to perform robustness-oriented finetuning.
We provide two example config files for finetuning: `finetune_vae.yaml` and `finetune_crop&resize.yaml`.
Key parameters to be aware of:

- **num_training_steps**: We recommend setting this between 20,000 and 50,000, depending on the scale and difficulty of your task.
- **ED_path**: Path to the pretrained model you want to finetune.
- **ft_noise_layers**: A list of distortion types you want the model to become robust against.
- **full_mask_ft**: Whether to use a global mask during finetuning. For example, for Crop & Resize, we enable this option since the distortion typically preserves only part of the watermarked image — similar to the behavior of a local mask.

This flexible finetuning setup allows for targeted robustness improvements tailored to your deployment environment.


## Cite
If you find this repository useful, please consider giving a star ⭐ and please cite as:
```
@article{hu2025mask,
  title={Mask Image Watermarking},
  author={Hu, Runyi and Zhang, Jie and Zhao, Shiqian and Lukas, Nils and Li, Jiwei and Guo, Qing and Qiu, Han and Zhang, Tianwei},
  journal={arXiv preprint arXiv:2504.12739},
  year={2025}
}
```
