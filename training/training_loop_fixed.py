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

def train_loop(args, accelerator, models, noise_scheduler, optimizer, lr_scheduler, train_dataloader, text_encoders, tokenizers):
    vae, unet = models["vae"], models["unet"]
    text_encoder_one, text_encoder_two = models["text_encoder_one"], models["text_encoder_two"]
    embedding_handler = models.get("embedding_handler")

    refer_model = copy.deepcopy(unet)
    refer_model.eval()
    for param in refer_model.parameters():
        param.requires_grad = False 

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

    model_outputs = deque(maxlen=100)  # 큐에서 저장할 최대 스텝 수를 100으로 유지

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                prompts = batch["prompts"]
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)

                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],)).long().to(latents.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(prompts, text_encoders, tokenizers, accelerator)
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}

                model_output = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample
                model_outputs.append(model_output.detach())  # 그래프에서 분리된 상태로 저장

                if len(model_outputs) >= 50 and (step + 1) % 100 == 0:  # 100 스텝마다 업데이트
                    losing_refer_output = model_outputs[0]  # 50 스텝 이전 값 사용
                else:
                    losing_refer_output = None  # 50 스텝 이전 값이 없다면 None으로 설정

                # 만약 losing_refer_output이 없으면 diff_lose를 0으로 설정
                if losing_refer_output is not None:
                    losing_noise = torch.randn_like(losing_refer_output)
                    losing_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (losing_refer_output.shape[0],)).long().to(losing_refer_output.device)
                    losing_noisy_latents = noise_scheduler.add_noise(losing_refer_output, losing_noise, losing_timesteps)

                    with torch.no_grad():
                        refer_output = refer_model(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample
                        losing_refer_output_updated = refer_model(losing_noisy_latents, losing_timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample

                    loss_model_lose = compute_loss(unet(losing_noisy_latents, losing_timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample, target_lose, noise_scheduler, losing_timesteps, losing_refer_output)

                    loss_refer_win = compute_loss(refer_output, target_win, noise_scheduler, timesteps, latents)
                    loss_refer_lose = compute_loss(losing_refer_output_updated, target_lose, noise_scheduler, losing_timesteps, losing_refer_output)

                    diff_lose = 0.3 * (loss_model_lose - loss_refer_lose)
                else:
                    # early steps where losing_refer_output is not available
                    diff_lose = 0

                target_win = noise
                target_lose = losing_noise if losing_refer_output is not None else None

                loss_model_win = compute_loss(model_output, target_win, noise_scheduler, timesteps, latents)

                if args.dcoloss > 0.0:
                    diff_win = loss_model_win - loss_refer_win if 'refer_output' in locals() else loss_model_win

                    diff = diff_win - diff_lose
                    inside_term = -1 * args.dcoloss * diff
                    loss = -1 * torch.nn.LogSigmoid()(inside_term)
                else:
                    loss = loss_model_win

                # Print loss and other values correctly
                print(f"Step: {step}, Loss: {loss.item() if isinstance(loss, torch.Tensor) else loss}, "
                      f"diff_win: {diff_win.item() if isinstance(diff_win, torch.Tensor) else diff_win}, "
                      f"diff_lose: {diff_lose.item() if isinstance(diff_lose, torch.Tensor) else diff_lose}, "
                      f"diff: {diff.item() if isinstance(diff, torch.Tensor) else diff}")

                accelerator.backward(loss)
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

        gc.collect()
        torch.cuda.empty_cache()

    progress_bar.finish()
    logger.info("Training completed")
