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

# from guppy import hpy; h=hpy()
# import pdb

bittensor.logging(debug = True)
wallet = bittensor.wallet()
subtensor = bittensor.subtensor()
meta = bittensor.metagraph()
meta.sync()
dend = bittensor.dendrite(wallet = wallet)
config = bittensor._neuron.text.core_validator.neuron.config()
config.nucleus.topk = 100
config.dataset.dataset_name = ['Books3']
config.dataset.num_batches = 100
print(config)
neuron = bittensor._neuron.text.core_validator.neuron(config = config)
nucleus = bittensor._neuron.text.core_validator.nucleus(config = config, device = 'cpu', subtensor = subtensor)
optimizer = torch.optim.SGD(
    nucleus.parameters(), lr=0.5, momentum=0.5
)

dataset = bittensor.dataset(config = config)
next(dataset)

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def display_top(snapshot, key_type='lineno', limit=100):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


bittensor.logging(debug = True)
# tracemalloc.start()

data1 = next(dataset)
count = 0 
random_uids = torch.tensor(list(range(100)))
random_endpoints = [meta.endpoints[uid] for uid in random_uids]
synapse = bittensor.synapse.TextCausalLMNext()

with torch.no_grad():
    query_responses, return_ops, times = dend.text(
        endpoints=random_endpoints,
        inputs=data1,
        synapses= [synapse],
        timeout=bittensor.__blocktime__
    )

endpoints, data = dend.format_text_inputs(random_endpoints, data1)
while count < 100*100:
    # === receptor pool only ===
    print('runnning')
    dend.receptor_pool.forward(endpoints, [synapse], data, 12)

    # === receptor only ===
    # for receptor in list(dend.receptor_pool.receptors.values()):
    #     receptor.forward([synapse], data1, 12)
    
    # pass
    # loss, stat = nucleus(data1, meta, dend)
    
    # if hasattr(loss, 'grad_fn') and loss.grad_fn is not None:
    #     loss.backward()

    # if count % 100 == 0:
    #     clip_grad_norm_(nucleus.parameters(), 0.5)
    #     optimizer.step()
    #     optimizer.zero_grad()

    count += 1
# nucleus(data2, meta, dend)
# nucleus(data3, meta, dend)
# nucleus(data3, meta, dend)
# nucleus(data3, meta, dend)
# nucleus(data3, meta, dend)
# nucleus(data3, meta, dend)
# nucleus(data3, meta, dend)

# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)
# print(h.heap())
# pdb.set_trace()