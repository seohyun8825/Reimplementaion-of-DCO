import torch
import torch.nn.functional as F
from safetensors.torch import save_file


class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.train_ids = None
        self.inserting_tokens = None
        self.embeddings_settings = {}

    def initialize_new_tokens(self, inserting_tokens, initializer_tokens):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            assert isinstance(inserting_tokens, list), "inserting_tokens should be a list of strings."
            assert all(
                isinstance(tok, str) for tok in inserting_tokens
            ), "All elements in inserting_tokens should be strings."

            self.inserting_tokens = inserting_tokens
            special_tokens_dict = {"additional_special_tokens": self.inserting_tokens}
            tokenizer.add_special_tokens(special_tokens_dict)
            text_encoder.resize_token_embeddings(len(tokenizer))

            self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_tokens)
            std_token_embedding = text_encoder.text_model.embeddings.token_embedding.weight.data.std()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding
            print(f"{idx} text encodedr's std_token_embedding: {std_token_embedding}")

            embeddings = []
            embeddings_norm = []
            for initializer_token in initializer_tokens:
                if initializer_token == "":
                    emb = torch.randn(1, text_encoder.text_model.config.hidden_size).to(device=self.device).to(dtype=self.dtype) * std_token_embedding
                    embeddings.append(emb)
                    embeddings_norm.append(std_token_embedding)
                else:
                    initializer_token_id = tokenizer.encode(initializer_token, add_special_tokens=False)
                    emb = text_encoder.text_model.embeddings.token_embedding.weight.data[initializer_token_id]
                    embeddings.append(emb)
                    embeddings_norm.append(emb.norm().item())
            
            embeddings = torch.cat(embeddings, dim=0)
            text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids] = embeddings
            embeddings_norm = torch.tensor(embeddings_norm).unsqueeze(1)
            self.embeddings_settings[f"token_embedding_norm_{idx}"] = embeddings_norm
    
            self.embeddings_settings[
                f"original_embeddings_{idx}"
            ] = text_encoder.text_model.embeddings.token_embedding.weight.data.clone()

            inu = torch.ones((len(tokenizer),), dtype=torch.bool)
            inu[self.train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = inu
            idx += 1

    def save_embeddings(self, file_path: str):
        assert self.train_ids is not None, "Initialize new tokens before saving embeddings."
        tensors = {}
        idx_to_text_encoder_name = {0: "clip_l", 1: "clip_g"}
        for idx, text_encoder in enumerate(self.text_encoders):
            assert text_encoder.text_model.embeddings.token_embedding.weight.data.shape[0] == len(
                self.tokenizers[0]
            ), "Tokenizers should be the same."
            new_token_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids]
            tensors[idx_to_text_encoder_name[idx]] = new_token_embeddings
        save_file(tensors, file_path)

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    @torch.no_grad()
    def retract_embeddings(self):
        for idx, text_encoder in enumerate(self.text_encoders):
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            text_encoder.text_model.embeddings.token_embedding.weight.data[index_no_updates] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            index_updates = ~index_no_updates
            new_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates]
            new_embeddings = F.normalize(new_embeddings, dim=-1) * self.embeddings_settings[f"token_embedding_norm_{idx}"].view(-1, 1).to(device=text_encoder.device)
            text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates] = new_embeddings.to(device=text_encoder.device).to(dtype=text_encoder.dtype)

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds
