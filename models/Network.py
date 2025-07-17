import lpips
import kornia
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from .Mask_Model import WatermarkModel


def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)


class MaskMark:

    def __init__(
        self, device, lr, num_training_steps=100000, encoder_weight=1, decoder_weight=10,
        model_config=None, mask_weight=0, use_scheduler="", **kwargs,
    ):
        # device
        self.device = device
        self.use_scheduler = use_scheduler

        # network
        self.encoder_decoder = WatermarkModel(**model_config).to(device)

        if hasattr(self.encoder_decoder.noise.noise, "list"):
            for model in self.encoder_decoder.noise.noise.list:
                model.to(device)
        else:
            self.encoder_decoder.noise.noise.to(device)

        # optimizer
        self.opt_encoder_decoder = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr
        )
        if self.use_scheduler:
            self.scheduler_encoder_decoder = get_scheduler(
                self.use_scheduler,
                optimizer=self.opt_encoder_decoder,
                num_warmup_steps=int(num_training_steps * 0.02),
                num_training_steps=num_training_steps,
            )

        # loss function
        self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
        self.criterion_MSE = nn.MSELoss().to(device)
        self.criterion_LPIPS = lpips.LPIPS(net='vgg').to(device)

        # weight of encoder-decoder loss
        self.encoder_weight = encoder_weight
        self.decoder_weight = decoder_weight
        self.mask_weight = mask_weight

    def train(
        self,
        images: torch.Tensor,
        messages: torch.Tensor,
        mask: torch.Tensor,
        use_jnd: bool = True,
    ):
        self.encoder_decoder.train()

        with torch.enable_grad():
            images, messages, mask = images.to(self.device), messages.to(self.device), mask.to(self.device)

            encoded_images, _, decoded_messages, mask_gt, mask_pd = \
                self.encoder_decoder(image=images, message=messages, mask=mask, use_jnd=use_jnd)

            '''
            train encoder and decoder
            '''
            self.opt_encoder_decoder.zero_grad()
            
            enc_loss = self.criterion_MSE(encoded_images, images)
            dec_loss = self.criterion_MSE(decoded_messages, messages) + self.mask_weight * self.criterion_MSE(mask_gt, mask_pd)
            total_loss = self.encoder_weight * enc_loss + self.decoder_weight * dec_loss
            total_loss.backward()

            self.opt_encoder_decoder.step()
            if self.use_scheduler:
                self.scheduler_encoder_decoder.step()

            # psnr
            psnr = -kornia.losses.psnr_loss(
                denormalize(encoded_images).detach() * mask,
                denormalize(images) * mask, 2
            )

            # ssim
            ssim = 1 - 2 * kornia.losses.ssim_loss(
                denormalize(encoded_images).detach() * mask,
                denormalize(images) * mask,
                window_size=5,
                reduction="mean"
            )

        '''
        decoded message error rate
        '''
        error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)
        mask_pred_acc = (mask_gt.gt(0.5) == mask_pd.gt(0.5)).float().mean().item()

        result = {
            "error_rate": error_rate,
            "mask_pred_acc": mask_pred_acc,
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "enc_loss": enc_loss.item(),
            "dec_loss": dec_loss.item(),
            "total_loss": total_loss.item()
        }
        return result

    def decoded_message_error_rate(self, message, decoded_message):
        length = message.shape[0]

        message = message.gt(0.5)
        decoded_message = decoded_message.gt(0.5)
        error_rate = float(sum(message != decoded_message)) / length
        return error_rate

    def decoded_message_error_rate_batch(self, messages, decoded_messages):
        error_rate = 0.0
        batch_size = len(messages)
        for i in range(batch_size):
            error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
        error_rate /= batch_size
        return error_rate

    def save_model(self, path_encoder_decoder: str):
        torch.save(self.encoder_decoder.state_dict(), path_encoder_decoder)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.load_state_dict(torch.load(path_encoder_decoder, map_location='cpu'), strict=False)
