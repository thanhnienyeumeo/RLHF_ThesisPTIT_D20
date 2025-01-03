import torch
#import dataloader
from torch.utils.data import DataLoader, Subset
from dataset import DahoasRMStaticDataset
from loss import KPairwiseLoss
# from src.trainers import RewardModelTrainer, FSDPRewardModelTrainer, AcceleratorRewardModelTrainer
import statistics
from gpt import GPTRewardModel, GPT
from configs import get_configs
import tiktoken
import click

def prepare_gpt2_input(prompt, device):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode, enc

def generate_gpt2(model, prompt, device, samples=2):
    model.eval()
    model.to(device)
    max_new_tokens = 512
    temperature = 1.0
    top_k = 50
    x, decode, enc = prepare_gpt2_input(prompt, device)

    for k in range(samples):
        y = model.generate(x,
                           max_new_tokens,
                           temperature=temperature,
                           top_k=top_k,
                        #    eos_token_id = enc.encode('\n\n')
                           )
        return decode(y[0].tolist())
    


def reply(model_name, prompt, device, model = None):
    prompt = "Human: " + prompt + "\n\nAssistant:"
    
    cfg = get_configs("gpt2-medium")
    if not model:
        if model_name == "raw":
            cfg.pretrain = "huggingface"
            model = GPT.from_pretrained(cfg)
        elif model_name == "sft":
            model = GPT.from_checkpoint(
                cfg,
        r".\runs\sft\sft_sft_202411102112_final.pt")
        else:
            model = GPT.from_checkpoint(
        cfg,
        r".\runs\ppo\ppo_ppo_202411150635_actor_final.pt")
    # prompt = "What is the meaning of life?"
    ans = generate_gpt2(model, prompt, device)
    #return ánss
    return ans