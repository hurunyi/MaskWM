import os
import argparse
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from models.Network import MaskMark
from models.Noise import Noise
from utils.loader import get_dataloader_segmentation
from utils.masks import get_mask_embedder


def main(args):
    set_seed(21)
    train_config = OmegaConf.load(args.train_config_path)
    model_config = OmegaConf.load(args.model_config_path)
    print(f"\nTrain config: {train_config}\n")
    print(f"\nModel config: {model_config}\n")

    result_folder = os.path.join("checkpoints", args.model_name, time.strftime("%Y_%m_%dT%H_%M_%S", time.localtime()))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    if not os.path.exists(result_folder + "models/"):
        os.makedirs(f"{result_folder}/models", exist_ok=True)
    with open(result_folder + "/train_log.txt", "w") as file:
        content = "-----------------------" + time.strftime(
            "Date: %Y/%m/%d %H:%M:%S", time.localtime()
        ) + "-----------------------\n"
        file.write(content)
    OmegaConf.save(train_config, os.path.join(result_folder, "train_config.yaml"))
    OmegaConf.save(model_config, os.path.join(result_folder, "model_config.yaml"))

    image_size, message_length = model_config["wm_enc_config"]["image_size"], model_config["wm_enc_config"]["message_length"]
    noise_layers = train_config["noise_layers"]
    ft_noise_layers = train_config["ft_noise_layers"]
    full_mask_ft = train_config["full_mask_ft"]
    train_config["noise_layers"] = "Identity()"
    batch_size, device, num_training_steps, num_save_steps = \
        train_config["batch_size"], train_config["device"], train_config["num_training_steps"], train_config["num_save_steps"]
    ED_path = train_config["ED_path"]
    dataset_path = train_config.pop("dataset_path")

    decoder_weight, decoder_weight_end = train_config["decoder_weight"], train_config["decoder_weight_end"]
    if decoder_weight != decoder_weight_end:
        decoder_weight_decay_step = (decoder_weight - decoder_weight_end) / (
            num_training_steps * train_config["num_decoder_weight_decay_ratio"]
        )
    else:
        decoder_weight_decay_step = 0

    train_dataloader = get_dataloader_segmentation(
        data_dir=os.path.join(dataset_path, "train2014"),
        ann_file=os.path.join(dataset_path, "annotations", "instances_train2014.json"),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        random_nb_object=True,
        multi_w=False,
        max_nb_masks=3
    )

    network = MaskMark(**train_config, model_config=model_config)
    mask_embedder = get_mask_embedder(**train_config["masks"])
    full_mask_embedder = get_mask_embedder(kind="full", invert_proba=1.0)

    if ED_path:
        is_ft = True
        network.load_model_ed(ED_path)
        num = int(ED_path.split("_")[-1].split(".")[0])
        network.decoder_weight = decoder_weight_end
    else:
        is_ft = False
        num = 0
        
    num_training_steps += num

    print(
        f'Encoder: {sum(p.numel() for p in network.encoder_decoder.encoder.parameters() if p.requires_grad) / 1e6:.1f}M parameters'
    )
    print(
        f'Decoder: {sum(p.numel() for p in network.encoder_decoder.decoder.parameters() if p.requires_grad) / 1e6:.1f}M parameters'
    )
    print("\nStart training : \n\n")

    finish_flag = False
    
    running_result = {
        "error_rate": 0.0,
        "mask_pred_acc": 0.0,
        "mask_percentage": 0.0,
        "decoder_weight": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "enc_loss": 0.0,
        "dec_loss": 0.0,
        "total_loss": 0.0,
    }

    start_time = time.time()
    add_noise = False
    add_all_mask = False
    
    while True:
        ''' train '''
        for images, masks in tqdm(train_dataloader):
            
            num += 1
            bsz, C, H, W = images.shape

            if num > 500 or is_ft:
                add_all_mask = True

            if (num > 1000 and not add_noise) or is_ft:
                network.encoder_decoder.noise = Noise(noise_layers)
                if hasattr(network.encoder_decoder.noise.noise, "list"):
                    for model in network.encoder_decoder.noise.noise.list:
                        model.to(device)
                else:
                    network.encoder_decoder.noise.noise.to(device)
                add_noise = True

            if num > num_training_steps * train_config["start_jnd_ratio"] or is_ft:
                use_jnd = True
            else:
                use_jnd = False

            message = torch.Tensor(np.random.choice([0, 1], (bsz, message_length)))
            
            if not add_all_mask:
                mask = full_mask_embedder(images)
                mask = torch.from_numpy(mask)
            else:
                mask = mask_embedder(images, masks=masks)

            if is_ft:
                if random.random() < 0.5:
                    used_noise_layers = ft_noise_layers
                    if full_mask_ft:
                        mask = full_mask_embedder(images)
                        mask = torch.from_numpy(mask)
                else:
                    used_noise_layers = noise_layers

                network.encoder_decoder.noise = Noise(used_noise_layers)
                if hasattr(network.encoder_decoder.noise.noise, "list"):
                    for model in network.encoder_decoder.noise.noise.list:
                            model.to(device)
                else:
                    network.encoder_decoder.noise.noise.to(device)

            result = network.train(images=images, messages=message, mask=mask, use_jnd=use_jnd)

            result["mask_percentage"] = mask.mean().item()
            result["decoder_weight"] = network.decoder_weight
            result["step"] = num

            if num % 10 == 0:
                print(f"\n{result}\n")

            for key in result:
                if key in running_result.keys():
                    running_result[key] += float(result[key])

            if network.decoder_weight > decoder_weight_end:
                network.decoder_weight -= decoder_weight_decay_step
            else:
                network.decoder_weight = decoder_weight_end

            if num % 1000 == 0:
                ''' train results '''
                content = "\nStep " + str(num) + " : " + str(int(time.time() - start_time)) + "\n"
                for key in running_result:
                    content += " " + key + "=" + str(running_result[key] / 1000) + "\n"

                with open(result_folder + "/train_log.txt", "a") as file:
                    file.write(content)
                print(content)

                running_result = {
                    "error_rate": 0.0,
                    "mask_pred_acc": 0.0,
                    "mask_percentage": 0.0,
                    "decoder_weight": 0.0,
                    "psnr": 0.0,
                    "ssim": 0.0,
                    "enc_loss": 0.0,
                    "dec_loss": 0.0,
                    "total_loss": 0.0,
                }
                start_time = time.time()

            if num % num_save_steps == 0:
                ''' save model '''
                path_model = f"{result_folder}/models"
                path_encoder_decoder = f"{path_model}/ckpt_{num}.pth"
                network.save_model(path_encoder_decoder)

            if num >= num_training_steps:
                finish_flag = True
                break

        if finish_flag:
            ''' save model '''
            path_model = f"{result_folder}/models"
            path_encoder_decoder = f"{path_model}/ckpt_{num}.pth"
            network.save_model(path_encoder_decoder)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train_config_path', type=str)
    parser.add_argument('--model_config_path', type=str)
    args = parser.parse_args()
    main(args)
