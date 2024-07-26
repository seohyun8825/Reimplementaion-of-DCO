import torch
import os
from PIL import Image
from safetensors.torch import load_file
from reward_guidance import RGPipe
def generate_image(checkpoint_dir, output_path, prompt, base_prompt, rg_scale=3.0, seed=42):
    # Initialize the pipe
    pipe = RGPipe.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")

    # Load lora weights and textual embeddings
    pipe.load_lora_weights(checkpoint_dir)

    inserting_tokens = ["<dog>"] # Load new tokens
    state_dict = load_file(os.path.join(checkpoint_dir, "learned_embeds.safetensors"))
    pipe.load_textual_inversion(state_dict["clip_l"], token=inserting_tokens, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
    pipe.load_textual_inversion(state_dict["clip_g"], token=inserting_tokens, text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

    # Set seed and generator
    generator = torch.Generator("cuda").manual_seed(seed)

    # Generate image
    if rg_scale > 0.0:
        image = pipe.my_gen(
            prompt=base_prompt,
            prompt_ti=prompt, 
            generator=generator,
            cross_attention_kwargs={"scale": 1.0},
            guidance_scale=7.5,
            guidance_scale_lora=rg_scale,
        ).images[0]
    else:
        image = pipe(
            prompt=prompt, 
            generator=generator,
            cross_attention_kwargs={"scale": 1.0},
            guidance_scale=7.5,
        ).images[0]

    # Convert the image to PIL format and save it
    image.save(output_path)
    print(f"Image saved at {output_path}")

# Define the prompts
prompt = "A <dog> playing piano" # Prompt including new tokens
base_prompt = "A dog playing piano" # Prompt without new tokens


generate_image("/home/user/Desktop/project/dco/output/checkpoint-5", "output_image_original.png", prompt, base_prompt)
#generate_image("checkpoint-900", "output_image_exp.png", prompt, base_prompt)