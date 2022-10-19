import argparse
import time
import datetime
import bittensor
import torch
import os
import wandb
import math
import random
import pandas
import traceback
from rich import print
from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.traceback import install
from typing import List, Tuple, Callable, Dict, Any, Union, Set
import sys
import tracemalloc
from collections import Counter
import linecache

from bittensor.utils.tokenizer_utils import phrase_cross_entropy

from torch.nn.utils import clip_grad_norm_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loguru import logger
from threading import Lock

bittensor.logging(debug = True)
wallet = bittensor.wallet()
subtensor = bittensor.subtensor(network = 'nobunaga')
meta = bittensor.metagraph(subtensor = subtensor)
meta.sync()
dend = bittensor.dendrite(wallet = wallet)
config = bittensor._neuron.text.core_validator.neuron.config()
config.nucleus.topk = 100
config.dataset.dataset_name = ['Books3']
config.dataset.num_batches = 100
dataset = bittensor.dataset(config = config)
next(dataset)

synapse = bittensor.synapse.TextCausalLMNext()
from bittensor._neuron.text.core_server.nucleus_impl import server
model = server()

def forward_casual_lm_next(inputs_x: torch.FloatTensor, model_output=None):
    message, model_output, topk_token_phrases = model.encode_forward_causallmnext(inputs_x.to(model.device),
                                                                                topk=4096,
                                                                                model_output=model_output)
    return message, model_output, topk_token_phrases

# === server forward ===
message, model_output, forward_response_tensor = forward_casual_lm_next(next(dataset))

# === server serialize === 
encoded_tensor = synapse.encode_forward_response_tensor ( forward_response_tensor )
tensor_serialzier = bittensor.serializer( serializer_type = synapse.forward_response_serializer_type )
forward_response_tensor = tensor_serialzier.serialize( tensor_obj = encoded_tensor, from_type = bittensor.proto.TensorType.TORCH )

# === validator deserialize ===
try:
    synapse.decode_forward_response_tensor(forward_response_tensor)
except Exception as e:
    print(e.detail)

print('done!')