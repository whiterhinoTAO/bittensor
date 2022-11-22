import pdb
import bittensor as bt

import torch
import time
import argparse

from transformers import AutoModelForCausalLM

from parallelformers import parallelize

# def split_models(model, num_gpus: int):
#     layers = model.transformer.h
#     layers_per_gpu = len(layers) // num_gpus
    
#     pdb.set_trace()

#     # move everything in the model (model.transformer) with the exception of the layers (model.transformer.h)

#     for module in model.transformer.children():
#         # check if the module is transformer.h
#         if module != model.transformer.h:
#             module = module.to('cuda:0')
        


#     # here we add the layers to the gpus
#     for i in range(len(layers)):
#         # assume the num_gpus is 4, and the layers_per_gpu is 3 (12 layers total)
#         # then the first 3 layers will be on gpu 0, the next 3 layers will be on gpu 1, etc.
#         gpu = i // layers_per_gpu
#         layer = layers[i]
#         device = torch.device(f"cuda:{gpu}")
#         layer.to(device)



#     return model



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_gpus', type=int, default=1)
#     args = parser.parse_args()

#     pre_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

#     tokenizer = bt.tokenizer()
#     model = split_models(pre_model, args.num_gpus)

#     inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda:0")
#     outputs = model(**inputs)

#     pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="EleutherAI/gpt-neo-125M")
    args = parser.parse_args()

    pre_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    tokenizer = bt.tokenizer()
    # parallelize(pre_model, num_gpus=args.num_gpus, fp16=True)
    pre_model = torch.nn.DataParallel(pre_model)
    pre_model.to("cuda")

    inputs = tokenizer("the dog is cute", return_tensors="pt").to("cuda")

    start_time = time.time()
    outputs = pre_model(**inputs)
    end_time = time.time()
    # target_output = 

    print(f"Time taken: {end_time - start_time}")
    pdb.set_trace()