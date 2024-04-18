#!/usr/bin/env python3

import copy
from dataclasses import asdict

import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, AutoencoderTiny, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from PIL.Image import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast
from tqdm import tqdm

from tld.denoiser import Denoiser
from tld.diffusion import DiffusionGenerator
from tld.configs import ModelConfig

logger = get_logger(__name__, log_level="INFO")

labels = [
    "A man in a suit",
    "A picture of a fiery orange and red mythical dragon-like figure, with smoke billowing from its nostrils.",
    "A cartoon image of a character that looks like a yellow sunflower with a smiling face in the center.",
    "An image of a serene mountain landscape at sunrise, with the sun casting golden hues over snow-capped peaks. Include a clear blue lake in the foreground reflecting the mountains and a sky filled with soft morning clouds.",
    "A sketch of a ghostly figure, transparent and white, with glowing red eyes and ethereal trails.",
    "A drawing of a cute, furry, brown bear cub-like character, with large, round ears and a small nose.",
    "A historical battle scene from the medieval period with knights in armor on horseback, charging across a grassy field. The background should show a large castle, spectators, and a dramatic, cloudy sky.",
    "An image of a bustling cityscape at night, illuminated by neon lights with reflections in a rain-soaked street. The scene includes people walking with umbrellas, a variety of glowing shop signs, and tall buildings."
]

# def log_validation(vae, text_encoder, tokenizer, unet, validation_prompts, accelerator, weight_dtype, epoch):
#     logger.info("Running validation... ")

#     pipeline = StableDiffusionPipeline(
#         vae=accelerator.unwrap_model(vae),
#         text_encoder=accelerator.unwrap_model(text_encoder),
#         tokenizer=tokenizer,
#         unet=accelerator.unwrap_model(unet),
#         safety_checker=None,
#         torch_dtype=weight_dtype,
#     )
#     pipeline = pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=True)
    
#     generator = torch.Generator(device=accelerator.device)

#     images = []
#     for i in range(len(validation_prompts)):
#         with torch.autocast("cuda"):
#             image = pipeline(validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

#         images.append(image)

#     for tracker in accelerator.trackers:
#         if tracker.name == "tensorboard":
#             np_images = np.stack([np.asarray(img) for img in images])
#             tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
#         elif tracker.name == "wandb":
#             tracker.log(
#                 {
#                     "validation": [
#                         wandb.Image(image, caption=f"{i}: {validation_prompts[i]}")
#                         for i, image in enumerate(images)
#                     ]
#                 }
#             )
#         else:
#             logger.warn(f"image logging not implemented for {tracker.name}")

#     del pipeline
#     torch.cuda.empty_cache()

#     return images

def eval_gen(unet, vae, tokenizer, text_encoder, scheduler, accelerator, labels) -> Image:
    # class_guidance = 4.5
    # seed = 10
    # out, _ = diffuser.generate(
    #     labels=torch.repeat_interleave(labels, 2, dim=0),
    #     num_imgs=16,
    #     seed=seed,
    #     n_iter=40,
    #     exponent=1,
    #     sharp_f=0.1,
    #     img_size=img_size
    # )
    
    pipeline = StableDiffusionPipeline(
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        scheduler = scheduler, 
        safety_checker=None, 
        feature_extractor = None
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)
    pipeline.save_pretrained(save_directory = '/w/340/ikozlov/trained_pipeline', push_to_hub=True, repo_id="ikozlov/MobileDiffusion")
    
    generator = torch.Generator(device=accelerator.device) #.manual_seed(seed)

    images = []
    for i in range(len(labels)):
        with torch.autocast("cuda"):
            image = pipeline(labels[i], num_inference_steps=20, generator=generator).images[0]

        images.append(to_tensor(image))
        
    out = torch.stack(images, axis=0)

    out = to_pil((vutils.make_grid((out + 1) / 2, nrow=8, padding=4)).float().clip(0, 1))
    out.save(f"emb_val_cfg.png")

    return out


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_per_layer(model: nn.Module):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")


to_pil = torchvision.transforms.ToPILImage()
to_tensor = torchvision.transforms.ToTensor()


def update_ema(ema_model: nn.Module, model: nn.Module, alpha: float = 0.999):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(model_param.data, alpha=1 - alpha)



def main(config: ModelConfig) -> None:
    """main train loop to be used with accelerate"""
    denoiser_config = config.denoiser_config
    train_config = config.train_config
    dataconfig = config.data_config

    log_with="wandb" if train_config.use_wandb else None
    accelerator = Accelerator(mixed_precision="fp16", log_with=log_with)

    accelerator.print("Loading Data:")
    latent_train_data = torch.tensor(np.load(dataconfig.latent_path)) #, dtype=torch.float16)
    train_label_embeddings = torch.tensor(np.load(dataconfig.text_emb_path)) # , dtype=torch.float16)
    # emb_val = torch.tensor(np.load(dataconfig.val_path), dtype=torch.float32)
    dataset = TensorDataset(latent_train_data, train_label_embeddings)
    train_loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    # vae = AutoencoderKL.from_pretrained(config.vae_cfg.vae_name, torch_dtype=config.vae_cfg.vae_dtype)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae") #, torch_dtype=config.vae_cfg.vae_dtype)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder") #, torch_dtype=torch.float16)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer") #, torch_dtype=torch.float16)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    teacher_model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet") #, torch_dtype=torch.float16)
    teacher_model.eval()
    noise_scheduler.config.prediction_type == "epsilon"

    if accelerator.is_main_process:
        vae = vae.to(accelerator.device)

    model = Denoiser(**asdict(denoiser_config))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=1e-7)

    if train_config.compile:
        accelerator.print("Compiling model:")
        model = torch.compile(model)
        teacher_model = torch.compile(teacher_model)

    if not train_config.from_scratch:
        accelerator.print("Loading Model:")
        wandb.restore(
            train_config.model_name, run_path=f"apapiu/cifar_diffusion/runs/{train_config.run_id}", replace=True
        )
        full_state_dict = torch.load(train_config.model_name)
        model.load_state_dict(full_state_dict["model_ema"])
        optimizer.load_state_dict(full_state_dict["opt_state"])
        global_step = full_state_dict["global_step"]
    else:
        global_step = 0

    if accelerator.is_local_main_process:
        ema_model = copy.deepcopy(model).to(accelerator.device)
        # diffuser = DiffusionGenerator(ema_model, vae, accelerator.device, torch.float32)

    accelerator.print("model prep")
    model, train_loader, optimizer, teacher_model = accelerator.prepare(model, train_loader, optimizer, teacher_model)

    if train_config.use_wandb:
        accelerator.init_trackers(project_name="cifar_diffusion", config=asdict(config))

    accelerator.print(count_parameters(model))
    accelerator.print(count_parameters_per_layer(model))

    ### Train:
    for i in range(1, train_config.n_epoch + 1):
        accelerator.print(f"epoch: {i}/{train_config.n_epoch}")

        for x, y in tqdm(train_loader):
            x = x / config.vae_cfg.vae_scale_factor
            
            # Sample a random timestep for each image
            bsz = x.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device)
            timesteps = timesteps.long()
            noise = torch.randn_like(x, device=accelerator.device)
            x_noisy = noise_scheduler.add_noise(x, noise, timesteps).float()

            # noise_level = torch.tensor(
            #     np.random.beta(train_config.beta_a, train_config.beta_b, len(x)), device=accelerator.device
            # )
            # signal_level = 1 - noise_level
            # noise = torch.randn_like(x)

            # x_noisy = noise_level.view(-1, 1, 1, 1) * noise + signal_level.view(-1, 1, 1, 1) * x

            # x_noisy = x_noisy.float()
            # noise_level = noise_level.float()
            label = y

            prob = 0.15
            mask = torch.rand(y.size(0), device=accelerator.device) < prob
            label[mask] = 0  # OR replacement_vector

            if global_step % train_config.save_and_eval_every_iters == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ##eval and saving:
                    # eval_gen(unet, vae, tokenizer, text_encoder, accelerator, labels)
                    out = eval_gen(model, vae, tokenizer, text_encoder, noise_scheduler, accelerator, labels)
                    out.save("img.jpg")
                    if train_config.use_wandb:
                        accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})

                    opt_unwrapped = accelerator.unwrap_model(optimizer)
                    full_state_dict = {
                        "model_ema": ema_model.state_dict(),
                        "opt_state": opt_unwrapped.state_dict(),
                        "global_step": global_step,
                    }
                    if train_config.save_model:
                        accelerator.save(full_state_dict, train_config.model_name)
                        if train_config.use_wandb:
                            wandb.save(train_config.model_name)

            model.train()

            with accelerator.accumulate():
                ###train loop:
                optimizer.zero_grad()

                # pred = model(x_noisy, noise_level.view(-1, 1), label)
                target = noise
                
                with autocast(): 
                    pred = model(x_noisy, timesteps, label).sample
                    loss_st = loss_fn(pred, target)
                    
                    with torch.no_grad():
                        teacher_pred = teacher_model(x_noisy, timesteps, label).sample
                    
                    loss_kd = loss_fn(pred, teacher_pred)
                    
                    alpha = 0.1
                    loss = alpha * loss_st + (1 - alpha) * loss_kd
                    
                    accelerator.log({"train_loss": loss.item()}, step=global_step)
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                if accelerator.is_main_process:
                    update_ema(ema_model, model, alpha=train_config.alpha)

            global_step += 1
            
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ##eval and saving:
        # eval_gen(unet, vae, tokenizer, text_encoder, accelerator, labels)
        out = eval_gen(model, vae, tokenizer, text_encoder, noise_scheduler, accelerator, labels)
        out.save("img.jpg")
        if train_config.use_wandb:
            accelerator.log({f"step: {global_step}": wandb.Image("img.jpg")})

        opt_unwrapped = accelerator.unwrap_model(optimizer)
        full_state_dict = {
            "model_ema": ema_model.state_dict(),
            "opt_state": opt_unwrapped.state_dict(),
            "global_step": global_step,
        }
        if train_config.save_model:
            accelerator.save(full_state_dict, train_config.model_name)
            if train_config.use_wandb:
                wandb.save(train_config.model_name)
     
    accelerator.end_training()



# args = (config, data_path, val_path)
# notebook_launcher(training_loop)
