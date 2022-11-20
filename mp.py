import pdb
import bittensor as bt

from transformers import AutoModelForCausalLM


# if __name__ == "__main__":
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

tokenizer = bt.tokenizer()

def split_models(model, num_gpus: int):
    layers = model.transformer.h
    layers_per_gpu = layers // num_gpus
    
    # a for loop that adds the layers to the gpu with .to(device)
    for i in range(len(layers)):
        # add the layer to the gpu
        layers[i].to(f"cuda:{i // layers_per_gpu}")

    pdb.set_trace()



if __name__ == "__main__":
    split_models(model, 1)