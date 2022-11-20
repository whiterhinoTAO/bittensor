import pdb
import bittensor as bt

import torch
import argparse

from transformers import AutoModelForCausalLM



def split_models(model, num_gpus: int):
    layers = model.transformer.h
    layers_per_gpu = len(layers) // num_gpus
    

    for i in range(len(layers)):
        # assume the num_gpus is 4, and the layers_per_gpu is 3 (12 layers total)
        # then the first 3 layers will be on gpu 0, the next 3 layers will be on gpu 1, etc.
        gpu = i // layers_per_gpu
        layer = layers[i]
        device = torch.device(f"cuda:{gpu}")
        layer.to(device)



    pdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    tokenizer = bt.tokenizer()
    split_models(model, args.num_gpus)