# Mask Image Watermarking

Official implementation of [Mask Image Watermarking](http://arxiv.org/abs/2504.12739).

## 🔗 Model Weights

We provide pre-trained model weights for inference. You can download them from the following link:

**[Download Model Weights]([https://example.com/download-link](https://huggingface.co/Runyi-Hu/MaskMark))**

After downloading, place the weights into the `checkpoints/` directory.

## 🚀 Inference

To run inference, use the following command:

```bash
python3 inference.py \
--device "cuda:0" \
--model_name "D_32bits" \
--image_name "00"
```

The generated results will be saved to the `results/` directory.

## 🛠️ Training
Training code will be released in a future update. Stay tuned!

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
