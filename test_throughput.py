import bittensor
import torch

graph = bittensor.metagraph().sync()
wallet = bittensor.wallet(name = 'default3', hotkey = 'default')
dend = bittensor.dendrite( wallet = wallet ) 

import time 
import psutil
import tqdm 
import random
start_time = time.time()
io_1 = psutil.net_io_counters()
start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv

bittensor.logging(debug = True)
def get_size(bytes):
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024

n_steps = 1
n_queried = 100
timeout = 9

inputs = torch.ones([10, 20], dtype = torch.int64) 

results = []
for step in range(n_steps):
    uids = random.sample( range(4096), n_queried )
    endpoints = graph.endpoints[uids]
    a, b, c = dend.text( endpoints=endpoints, synapses=[bittensor.synapse.TextCausalLM()], inputs=inputs, timeout = timeout)
    results.append( [bi.item() == 1 for bi in b])

io_2 = psutil.net_io_counters()
total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv
end_time = time.time()

total_success = sum([sum(ri) for ri in results])
total_sent = n_queried * n_steps
total_failed = total_sent - total_success
total_seconds =  end_time - start_time

print ('\nTotal:', total_sent, 
       '\nSteps:', n_steps, 
       '\nQueries:', n_queried,
       '\nTimeout:', timeout,
       '\nSuccess:', total_success, 
       '\nFailures:', total_failed, 
       '\nRate:', total_success/total_sent, 
       '\nSize:', list(inputs.shape), 
       '\nSeconds:', total_seconds, '/s',
       '\nQ/sec:', total_success/total_seconds, '/s',
       '\nTotal Upload:', get_size( total_bytes_sent ),
       '\nTotal Download:', get_size( total_bytes_recved ),
       '\nUpload Speed:', get_size( total_bytes_sent / total_seconds), "/s",
       '\nDownload Speed:', get_size( total_bytes_recved / total_seconds), "/s")
