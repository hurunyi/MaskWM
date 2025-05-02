import os
import numpy as np
import torch
import argparse
from PIL import Image
from omegaconf import OmegaConf
from kornia.metrics import psnr, ssim
from models.Mask_Model import WatermarkModel
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)


def decoded_message_accuracy(message, decoded_message):
    message = message.gt(0.5)
    decoded_message = decoded_message.gt(0.5)
    accuracy = (message == decoded_message).float().mean().item()
    return accuracy


def calculate_iou_score(preds, targets):
    preds = preds.int()
    targets = targets.int()

    tp = torch.sum((preds == 1) & (targets == 1)).float()
    fp = torch.sum((preds == 1) & (targets == 0)).float()
    fn = torch.sum((preds == 0) & (targets == 1)).float()
    tn = torch.sum((preds == 0) & (targets == 0)).float()

    iou_1 = (tp / (tp + fp + fn + 1e-8)).item()
    iou_0 = (tn / (tn + fp + fn + 1e-8)).item()
    
    return iou_1, iou_0


@torch.no_grad()
def main(args):
    device = args.device
    if "ED_" in args.model_name:
        jnd_factor = 1.75
        model_config = OmegaConf.load("configs/model/ED_32bits.yaml")
        embedding_mask = 1
    else:
        jnd_factor = 1.3
        model_config = OmegaConf.load("configs/model/D_32bits.yaml")
        embedding_mask = 0
    
    image_size = 512
    H, W = image_size, image_size
    message_length = model_config["wm_enc_config"]["message_length"]

    transform = Compose([        
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    mask_transform = Compose([        
        Resize(image_size, interpolation=InterpolationMode.NEAREST),
        CenterCrop(image_size),
        ToTensor()
    ])

    wm_model = WatermarkModel(**model_config)
    ckpt_path = f'checkpoints/{args.model_name}.pth'
    wm_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
    wm_model = wm_model.to(device)
    wm_model.eval()
    
    image_path = f'examples/images/{args.image_name}.png'
    mask_path = f'examples/masks/{args.image_name}.png'
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    image = transform(image).unsqueeze(0).to(device)
    mask = mask_transform(mask).unsqueeze(0).to(device)
    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

    image_256 = F.interpolate(image, size=[256, 256], mode="bilinear")
    mask_256 = F.interpolate(mask, size=[256, 256], mode="bilinear")

    if embedding_mask:
        wm_image_256 = wm_model.encoder(image_256, message, mask_256[:,:1], jnd_factor=jnd_factor, blue=True)
    else:
        wm_image_256 = wm_model.encoder(image_256, message, jnd_factor=jnd_factor, blue=True)
    wm_image = (F.interpolate((wm_image_256-image_256), size=[H, W], mode="bilinear") + image).clamp_(-1, 1)
    
    psnr_value = psnr(denormalize(wm_image), denormalize(image), 1).item()
    ssim_value = torch.mean(ssim(denormalize(wm_image), denormalize(image), window_size=11)).item()

    wm_image_fused = wm_image * mask + image * (1 - mask)
    wm_image_fused_256 = TF.resize(wm_image_fused, [256, 256])
    
    decoded_message, mask_pred_256 = wm_model.decoder(wm_image_fused_256)
    mask_pred = F.interpolate(mask_pred_256, size=[H, W], mode="bilinear")
    threshold = 0.5
    mask_pred = (mask_pred > threshold).float()

    acc = decoded_message_accuracy(message, decoded_message)
    iou_1, iou_0 = calculate_iou_score(mask_pred, mask)

    save_img_dir = f"./results/{args.model_name}"
    os.makedirs(save_img_dir, exist_ok=True)
    
    img_orig = TF.to_pil_image((image[0] + 1) / 2)
    img_wm = TF.to_pil_image((wm_image[0] + 1) / 2)
    img_res = TF.to_pil_image(torch.abs(wm_image - image)[0] * 10)
    mask_pred_img = TF.to_pil_image(mask_pred[0])

    img_orig.save(f"{save_img_dir}/orig_{args.image_name}.png")
    img_wm.save(f"{save_img_dir}/wm_{args.image_name}.png")
    img_res.save(f"{save_img_dir}/res_{args.image_name}.png")
    mask_pred_img.save(f"{save_img_dir}/mask_pd_{args.image_name}.png")
    
    print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, Bit Accuracy: {acc:.4f}, Watermarked IoU: {iou_1:.4f}, Unwatermarked IoU: {iou_0:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--image_name', type=str)
    args = parser.parse_args()
    main(args)
