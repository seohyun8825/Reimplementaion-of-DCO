import torch
import gc
import logging
import copy
import progressbar
from collections import deque
from training.utils import compute_text_embeddings, compute_time_ids
from training.checkpoint import save_checkpoint
from training.loss import compute_loss
from training.hooks import register_hooks

logger = logging.getLogger(__name__)


###################################################################################################################################################################
#loosing 데이터가 5번정도 초반에만 업데이트 결과 in detailed_prompt_2000_losedatamovebut5update
#만약에 lose data를 계속 업데이트하면, 그냥 sample 데이터랑 아예 똑같아져버림(너무 많이 학습시켜서 그런걸수있는데) 이거 결과는4000_lr5e_80que_30mean_loosing_data_move 여기
#배경까지 똑같알져버림..
###############################################################################################################################################################

def train_loop(args, accelerator, models, noise_scheduler, optimizer, lr_scheduler, train_dataloader, text_encoders, tokenizers):
    vae, unet = models["vae"], models["unet"]
    text_encoder_one, text_encoder_two = models["text_encoder_one"], models["text_encoder_two"]
    embedding_handler = models.get("embedding_handler")

    refer_model = copy.deepcopy(unet)
    for param in refer_model.parameters():
        param.requires_grad = False  # Freeze the reference model

    global_step, first_epoch = 0, 0

    logger.info("Start")

    time_ids = compute_time_ids(args, accelerator)
    register_hooks(accelerator, models, args)

    widgets = [
        ' [', progressbar.Percentage(), '] ',
        progressbar.Bar(), ' (', progressbar.ETA(), ') ',
        progressbar.DynamicMessage('loss')
    ]
    progress_bar = progressbar.ProgressBar(max_value=args.max_train_steps, widgets=widgets)
    progress_bar.start()

    previous_outputs = deque(maxlen=50)  
    noise_strength = 0.0 
    update_count = 0  
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                try:
                    if global_step < args.max_train_steps * 0.01:  
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = args.learning_rate

                    prompts = batch["prompts"]
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],)).long().to(latents.device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(prompts, text_encoders, tokenizers, accelerator)
                    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

                    model_output = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample

                    with torch.no_grad():
                        refer_output = refer_model(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample

                    target = noise

                    loss_model = compute_loss(model_output, target, noise_scheduler, timesteps, latents)

                    if args.dcoloss > 0.0:
                        loss_refer = compute_loss(refer_output, target, noise_scheduler, timesteps, latents)

                        if global_step < args.max_train_steps * 0.01:
                            loosing_output = refer_output
                        else:

                            if len(previous_outputs) == 50 and update_count < 5:  
                                loosing_output = torch.mean(torch.stack(list(previous_outputs)[-50:]), dim=0) 
                                update_count += 1  
                            else:
                                loosing_output = refer_output 

                        if global_step >= args.max_train_steps * 0.01 and global_step < args.max_train_steps * 0.5:
                            noise_for_loosing = torch.randn_like(loosing_output) * noise_strength
                            loosing_output = loosing_output + noise_for_loosing
                            noise_strength = max(0.1, noise_strength * 0.9) 

                        loss_loosing = compute_loss(loosing_output, target, noise_scheduler, timesteps, latents)

                        diff_winning = loss_model - loss_refer
                        diff_loosing = loss_loosing - loss_refer

                        if torch.abs(diff_loosing) < 1e-8:
                            diff_loosing = 0.0

                        inside_term = -1 * args.dcoloss * (diff_winning - 0.007 * diff_loosing)
                        loss = -1 * torch.nn.LogSigmoid()(inside_term)


                        previous_outputs.append(model_output.detach())

                    else:
                        loss = loss_model

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)  

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        global_step += 1
                        progress_bar.update(global_step, loss=loss.item())

                        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                        accelerator.log(logs, step=global_step)

                        if global_step % args.checkpoint_save == 0:
                            save_checkpoint(accelerator, args, unet, text_encoder_one, text_encoder_two, embedding_handler, global_step)

                    if global_step >= args.max_train_steps:
                        break

                except Exception as e:
                    logger.error(f"Error at epoch {epoch}, step {step}: {e}")
                    raise

        gc.collect()
        torch.cuda.empty_cache()

    progress_bar.finish()
    logger.info("Training completed")
