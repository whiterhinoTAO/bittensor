import os
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '0'))
device = torch.device("cuda", local_rank)


model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

ds_engine = deepspeed.init_inference(model,
                         mp_size=world_size,
                         dtype=torch.half,
                        #  replace_method='auto',
                        #  replace_with_kernel_inject=True
                         )

model_engine = ds_engine.module

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:

    input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors='pt').to(device)
    outputs = model_engine.generate(input_ids, max_length=128, do_sample=True)
    outputs
    import code; code.interact(local=dict(globals(), **locals()))