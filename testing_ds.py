import os
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import json


world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))
device = torch.device("cuda", local_rank)


def simple_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='DS config file.', default='/home/ubuntu/.bittensor/bittensor/bittensor/_neuron/text/core_server/ds_config.json')
    parser.add_argument('--local_rank', type=int, help='local rank', default=0)
    args = parser.parse_args()
    args.deepspeed_config = args.config_file
    args.config = json.load(open(args.config_file, 'r', encoding='utf-8'))
    return args



model = AutoModelForCausalLM.from_pretrained("gpt2-large")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

ds_args = simple_args()

model_engine, optimizer, _, _ = deepspeed.initialize(
    args = ds_args,
    model = model,
    # model = self.net,
    model_parameters = model.parameters(),
    # training_data = self.dataset
)

# ds_engine = deepspeed.init_inference(model,
#                          mp_size=world_size,
#                          dtype=torch.float,
#                         #  replace_method='auto',
#                         #  replace_with_kernel_inject=True
#                          )

# model_engine = ds_engine.module

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

    input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors='pt').to(device)
    outputs = model_engine.generate(input_ids, max_length=128, do_sample=True)
    print(outputs)
    import code; code.interact(local=dict(globals(), **locals()))