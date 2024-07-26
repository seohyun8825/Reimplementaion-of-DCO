import torch
from tqdm.auto import tqdm
import gc
import logging

from training.utils import compute_text_embeddings, compute_time_ids
from training.checkpoint import save_checkpoint
from training.loss import compute_loss
from training.hooks import register_hooks

logger = logging.getLogger(__name__)

def train_loop(args, accelerator, models, noise_scheduler, optimizer, lr_scheduler, train_dataloader, text_encoders, tokenizers):
    vae, unet = models["vae"], models["unet"]
    text_encoder_one, text_encoder_two = models["text_encoder_one"], models["text_encoder_two"]
    embedding_handler = models.get("embedding_handler")
    global_step, first_epoch = 0, 0

    logger.info("Starting training loop")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    time_ids = compute_time_ids(args, accelerator)
    register_hooks(accelerator, models, args)

    for epoch in range(first_epoch, args.num_train_epochs):
        #logger.info(f"Epoch {epoch} start")
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                try:
                    #logger.info(f"Epoch {epoch}, Step {step}: Loading and preparing data")
                    prompts = batch["prompts"]
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],)).long().to(latents.device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(prompts, text_encoders, tokenizers, accelerator)
                    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
                    model_output = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample

                    target = noise
                    loss = compute_loss(model_output, target, noise_scheduler, timesteps, latents)
                    #logger.info(f"Epoch {epoch}, Step {step}: Computing gradients")
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)

                    #logger.info(f"Epoch {epoch}, Step {step}: Updating optimizer and learning rate scheduler")
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                        progress_bar.set_postfix(**logs)
                        accelerator.log(logs, step=global_step)

                        if global_step % args.checkpoint_save == 0:
                            save_checkpoint(accelerator, args, unet, text_encoder_one, text_encoder_two, embedding_handler, global_step)

                    if global_step >= args.max_train_steps:
                        break

                except Exception as e:
                    logger.error(f"Error at epoch {epoch}, step {step}: {e}")
                    raise

        #logger.info(f"Epoch {epoch} end")
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Training completed")
