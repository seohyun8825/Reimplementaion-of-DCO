import torch
import gc
import logging
import copy
import progressbar
from training.utils import compute_text_embeddings, compute_time_ids
from training.checkpoint import save_checkpoint
from training.loss import compute_loss
from training.hooks import register_hooks

logger = logging.getLogger(__name__)

def train_loop(args, accelerator, models, noise_scheduler, optimizer, lr_scheduler, train_dataloader, text_encoders, tokenizers):
    vae, unet = models["vae"], models["unet"]
    text_encoder_one, text_encoder_two = models["text_encoder_one"], models["text_encoder_two"]
    embedding_handler = models.get("embedding_handler")

    #unet복사
    refer_model = copy.deepcopy(unet)
    for param in refer_model.parameters():
        param.requires_grad = False  # reference 모델은 freeze

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

    previous_model_output = None 
    noise_strength = 1.0 

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                try:
                    # adaptijve lr
                    if global_step < args.max_train_steps * 0.1:  # 학습 초기 10% 동안
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
                    #이걸하면 안될거같은데 좀이따 주석처리해서 실험해보기
                    model_output = model_output + 0.05 * torch.randn_like(model_output)  # 노이즈 강도를 증가

                    with torch.no_grad():
                        refer_output = refer_model(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample

                    target = noise

                    loss_model = compute_loss(model_output, target, noise_scheduler, timesteps, latents)

                    if args.dcoloss > 0.0:
                        # DCO Loss 계산
                        loss_refer = compute_loss(refer_output, target, noise_scheduler, timesteps, latents)

                        # Loosing 데이터가 이전 스텝에서 나온 output
                        if previous_model_output is not None:
                            loosing_output = previous_model_output
                        else:
                            loosing_output = refer_output  

                        # 근데 loosing 데이터가 처음에는 너무 똑같으니까 살짝 perturb
                        if global_step < args.max_train_steps * 0.5:  # 학습 초반 50% 동안 
                            noise_for_loosing = torch.randn_like(loosing_output) * noise_strength
                            loosing_output = loosing_output + noise_for_loosing
                            noise_strength = max(0.1, noise_strength * 0.95)  # 녿이즈 강도를 천천히 줄여보기

                        loss_loosing = compute_loss(loosing_output, target, noise_scheduler, timesteps, latents)

                        # 기존 DCO 에서 제안한 term이 diff winning, diff_loosing은 새로 정의한거
                        diff_winning = loss_model - loss_refer
                        diff_loosing = loss_loosing - loss_refer

                        # P
                        #print(f"Step {step}:")
                        #print(f"  loss_model: {loss_model.item()}")
                        #print(f"  loss_refer: {loss_refer.item()}")
                        #print(f"  loss_loosing: {loss_loosing.item()}")
                        #print(f"  diff_winning: {diff_winning.item()}")
                        #print(f"  diff_loosing: {diff_loosing.item()}")


                        if torch.abs(diff_loosing) < 1e-8:
                            diff_loosing = 0.0
                        #근데 왜 diff winning diff loosing 숫자 차이가 너무 큼. diff loosing 이 너무 영향이 세서, 거의 1000배 크게 나옴. 0.001 배해보자
                        inside_term = -1 * args.dcoloss * (diff_winning - 0.001*diff_loosing)
                        loss = -1 * torch.nn.LogSigmoid()(inside_term)

                        #print(f"  inside_term: {inside_term.item()}")
                        #print(f"  final_loss: {loss.item()}")

                        # 이전 모델의 출력을 저장하여 다음 스텝에서 loosing 데이터로 사용
                        previous_model_output = model_output.detach()
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
