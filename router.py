##################
##### Import #####
##################
import torch
import concurrent.futures
import time
import psutil
import math
import random
import argparse
import bittensor
import gc
from tqdm import tqdm
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from bittensor._neuron.text.neuron_utilities import ThreadQueue, PositionalEncoding, calc_loss_fct
from rich.traceback import install
import pdb
from bittensor._neuron.text.core_server.nucleus_impl import server
from bittensor.utils.tokenizer_utils import phrase_cross_entropy
import numpy as np
import pandas as pd
import time

import cProfile, pstats, io
from pstats import SortKey

install(show_locals=False)
import wandb 

###################
##### Helpers #####
###################
def get_size(bytes):
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024
        
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


#########################
##### Build Network #####
#########################
class Nucleus(nn.Module):
    def __init__(self, config, wallet, graph ):
        super(Nucleus, self).__init__()
        self.config = config
        self.wallet = wallet
        # self.graph = graph
        self.sigmoid = torch.nn.Sigmoid()
        self.synapse = bittensor.TextCausalLMNext()
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.tokenizer = bittensor.tokenizer()
        self.pad_token = self.tokenizer(self.tokenizer.pad_token)['input_ids'][0]
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.receptors = [bittensor.receptor( wallet=self.wallet, endpoint = graph.endpoint_objs[i] ) for i in range(graph.n)]
        self.model_baseline = server(config = config, model_name = 'EleutherAI/gpt-neo-125M')
        self.loss_fnc = torch.nn.CrossEntropyLoss()

        self.token_embedding = torch.nn.Embedding( 
            bittensor.__vocab_size__,  
            bittensor.__network_dim__ 
        )
        self.local_pos_encoder = PositionalEncoding( 
            bittensor.__network_dim__, 
            0.8 
        )
        self.gates = torch.nn.Linear( 
            bittensor.__network_dim__, 
            4096, bias=True 
        )
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer( 
                bittensor.__network_dim__, 
                config.nucleus.nhead, 
                config.nucleus.nhid, 
                config.nucleus.dropout, 
                batch_first=True
            ),
            config.nucleus.nlayers
        )
    
    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200 )
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default = 2 )
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2 )
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.timeout',type=int, default=4, help='''Timeout for each set of queries (we always wait this long)''')
        parser.add_argument('--nucleus.n_queried', type=int, default=100, help='''The number of endpoints we query each step.''')
        parser.add_argument('--nucleus.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--nucleus.validation_len', type=int, help='the number of tokens to validate', default=8 )
        parser.add_argument('--nucleus.sim_network_size', type=int, help='the number of uids to validate', default=200 )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        return bittensor.config( parser )

    def cal_loss(self, inputs, query_response, validation_len = 8, p = False):
        inputs_nxt = inputs[..., -validation_len:]
        _losses_val, _losses, val_probs = phrase_cross_entropy(inputs_nxt, query_response, reduce=True, p = p)
        return _losses_val, val_probs
    
    def sort_response_by_token(self, response, batch_size = None, all_logits = None):

        if batch_size == None:
            batch_size = response.shape[0]
        if all_logits == None:
            all_logits = torch.tensor(list(range(bittensor.__vocab_size__)))
            
        response_sort = torch.zeros(batch_size, bittensor.__vocab_size__ , 2).to(self.config.nucleus.device)
        response_sort[:, :, 1] = all_logits.repeat(batch_size, 1)

        for seq in range(batch_size):
            r_batch = response[seq, : -1, :] # [seq, tokens, prob]  
            index = r_batch[:, 1].long().to(self.config.nucleus.device)
            prob = r_batch[:, 0].to(self.config.nucleus.device)
            # unique_labels, labels_count = index.unique(return_counts=True)

            response_sort[seq, :, 0] = response_sort[seq, :, 0].scatter_add(0, index, prob) # to(self.config.nucleus.device)
            # div = torch.ones(bittensor.__vocab_size__, dtype=torch.long).to(self.config.nucleus.device)
            # div[unique_labels] = labels_count
            # response_sort[seq, :, 0] /= div

        return response_sort

    def mix_response(self, response_success, normalized_topk_routing_scores):
        batch_size = response_success[0][0].shape[0]
        print('response shape', response_success[0][0].shape)
        topk =  response_success[0][0].shape[1] - 1
        mixed_response = torch.zeros(batch_size, bittensor.__vocab_size__ + 1  , 2).to(self.config.nucleus.device)
        all_logits = torch.tensor(list(range(bittensor.__vocab_size__)))
        mixed_response[:, : -1, 1] = all_logits.repeat(batch_size, 1)


        for r, w in list(zip(response_success, normalized_topk_routing_scores)):
            response_sorted = self.sort_response_by_token(r[0], batch_size, all_logits)
            mixed_response[:, :-1, 0] += w.to(self.config.nucleus.device) * response_sorted[:, :, 0]

        for batch in range(batch_size):
            # mixed_batch_topk = mixed_response[batch, :, :][mixed_response[batch, :, 0].sort(descending=True)[1]][:topk, :]
            # floor_prob = (1 - sum(mixed_batch_topk[:, 0])) / (bittensor.__vocab_size__ - topk)
            # mixed_response[batch, :topk, :] = mixed_batch_topk
            # mixed_response[batch, topk:, :] = torch.zeros_like(mixed_response[batch, topk:, :])
            # mixed_response[batch, -1, :] = torch.tensor([[floor_prob, -1]])
            mixed_response[batch, -1, :] = torch.tensor([[0, -1]])

        # return reduced_mixed_response
        return mixed_response

    def forward(self, uids, inputs, dendrite):
        inputs = inputs.to(self.config.nucleus.device)
        inputs_seq = inputs[..., :-self.config.nucleus.validation_len] 
        start_time = time.time()
        
        # Route
        embedding = self.token_embedding(inputs) * math.sqrt(bittensor.__network_dim__)
        src_mask = torch.triu(torch.ones(embedding.size(1), embedding.size(1)) * float('-inf'), diagonal=1).to(self.config.nucleus.device)
        pos_embedding = self.local_pos_encoder(embedding)
        routing_context = self.encoder(pos_embedding, mask=src_mask)
        routing_score = torch.mean(self.sigmoid(self.gates(routing_context[:, -1, :])), dim=0)
        
        # Query
        endpoints = [graph.endpoints[uid] for uid in uids]
        topk_routing_scores = routing_score[uids]

        with torch.no_grad():
            start_time = time.time()
            responses, return_ops, times = dendrite.text(
                    endpoints=endpoints,
                    inputs=inputs_seq,
                    synapses=[self.synapse],
                    timeout=self.config.nucleus.timeout
                )

            dend_forward_time = time.time()
            qps = sum([ops == 1 for ops in return_ops]) / (time.time() - start_time)
        
        # Join responses
        successes = [ op.item() == 1 for op in return_ops]
        if sum(successes) == 0:
            stats = {
                'loss/min': torch.nan,
                'loss/average': torch.nan,
                'loss/routing': torch.nan,
                'loss/routing_baseline': torch.nan,
                'loss/model_baseline': torch.nan,
                'stat/receptor_time_avg': torch.nan,
                'stat/dend_time': dend_forward_time - start_time,
                'stat/batch_time': time.time() - start_time,
                'stat/qps': qps,
            }
            return stats, successes, routing_score, []
            
        for i, (r, op) in enumerate(list(zip(responses, return_ops))):
            if op == 1 and r[0][:, :, 0].sum() < 0:
                return_ops[i] = 0

        if 3566 in uids.tolist():
            idx = uids.tolist().index(3566)
            responses_gptj = [responses[idx]]
            return_ops_gptj = [return_ops[idx]]
            times_gptj = [times[idx]]
        
        uid_success = [uid for uid, op in list(zip(uids, return_ops)) if op == 1]
        response_success = [(r[0].to(self.config.nucleus.device),) for r, op in list(zip(responses, return_ops)) if op == 1]
        time_success = [t for t, op in list(zip(times, return_ops)) if op == 1]

        normalized_topk_routing_scores = topk_routing_scores[successes]/topk_routing_scores[successes].sum().to(self.config.nucleus.device)
        mixed_response = self.mix_response(response_success, normalized_topk_routing_scores)
        averaged_response = self.mix_response(response_success, torch.ones_like(normalized_topk_routing_scores) / len(normalized_topk_routing_scores))
        
        # Compute loss.
        loss_routing, _ = self.cal_loss(inputs, mixed_response, self.config.nucleus.validation_len)
        loss_routing_baseline, _ = self.cal_loss(inputs, averaged_response, self.config.nucleus.validation_len)
        losses = torch.tensor([self.cal_loss(inputs, r[0], self.config.nucleus.validation_len)[0] for r in response_success])
        
        print('check num uids and losses', len(uids), len(response_success), len(losses))
        top_mix_loss = {}
        for i in range(1, min(11, len(response_success)), 2):
            top_mix_response = self.mix_response( [response_success[idx] for idx in losses.sort()[1][:i]], torch.ones(i) / i)
            top_uid = [uid_success[idx] for idx in losses.sort()[1][:i]]
            print(f'===== top_mix{i} ======\n', top_uid)
            loss_top_mix, _ = self.cal_loss(inputs, top_mix_response, self.config.nucleus.validation_len, p = False)
            print('loss: ', loss_top_mix)
            top_mix_loss[f'loss/top_mix_{i}'] = loss_top_mix

        # for i in range(1, min(11, len(response_success)), 2):       
        #     org_response = response_success[losses.sort()[1][i]]
        #     org_response_sorted = self.sort_response_by_token(org_response[0])
        #     top_mix_response = self.mix_response([org_response], torch.ones(1))
        #     top_uid = [uid_success[losses.sort()[1][i]]]
        #     print(f'===== loss diff{i} ======\n', top_uid)
        #     loss_org, check1 = self.cal_loss(inputs, org_response[0], self.config.nucleus.validation_len, p = False)
        #     loss_org_sorted, check2 = self.cal_loss(inputs, org_response_sorted, self.config.nucleus.validation_len, p = False)
        #     loss_top_mix, _ = self.cal_loss(inputs, top_mix_response, self.config.nucleus.validation_len, p = False)
        #     print('loss: ', losses.sort()[0][i], loss_org, loss_org_sorted, loss_top_mix)
            
        #     for i, diff in enumerate(check2[0] - check1[0]):
        #         if diff > 0.05:
        #             print(check1[1][i])
        #             print(check2[1][i])
        
        stats = {
            'loss/min': losses.min(),
            'loss/count': len(losses),
            'loss/average': sum(losses) / len(losses),
            'loss/routing': loss_routing,
            'loss/routing_baseline': loss_routing_baseline,
            # 'loss/model_baseline': loss_model_baseline,
            'stat/receptor_time_avg': sum(time_success) / sum(successes),
            'stat/dend_time': dend_forward_time - start_time,
            'stat/batch_time': time.time() - start_time,
            'stat/qps': qps,
        }
        if 3566 in uids.tolist() and return_ops_gptj[0] == 1:
            gptj_response = self.mix_response( responses_gptj , torch.ones(1))
            stats['loss/gptj'] = self.cal_loss(inputs, gptj_response, self.config.nucleus.validation_len)[0]
        
        stats = {**stats, **top_mix_loss}
        stats = {**stats}
        
        # Return.
        return stats, successes, routing_score, [] #losses
    
    
##############################
##### Build config ###########
##############################
parser = argparse.ArgumentParser( 
    description=f"Bittensor Validator Training ",
    usage="python3 train.py <command args>",
    add_help=True
)
parser.add_argument( '--max_workers', type=int, default=10, help='''Maximum concurrent workers on threadpool''')
# parser.add_argument( '--n_steps', type=int, default=10, help='''The number of steps we run.''')
parser.add_argument( '--chunk_size', type=int, default=10, help='''The number of concurrent steps we run.''')
parser.add_argument( '--learning_rate', type=float, help='Training initial learning rate.', default=0.01)
parser.add_argument( '--momentum', type=float, help='optimizer momentum.', default=0.8)
parser.add_argument( '--use_wandb', action='store_true', default=False, help='''To use wandb to track results''')

bittensor.wallet.add_args(parser)
bittensor.logging.add_args(parser)
bittensor.subtensor.add_args(parser)
bittensor.dataset.add_args(parser)
bittensor.dendrite.add_args(parser)
bittensor.wandb.add_args(parser)
server.add_args(parser)
Nucleus.add_args(parser)
config = bittensor.config(parser = parser)
print (config)

##########################
##### Setup objects ######
##########################
# Sync graph and load power wallet.
bittensor.logging( config = config )
dataset = bittensor.dataset( config = config )
got_sub = False
while not got_sub:
    try:
        subtensor = bittensor.subtensor( config = config )
        got_sub = True
    except:
        time.sleep(1)
graph = bittensor.metagraph( subtensor = subtensor ).sync()
wallet = bittensor.wallet( config = config )
dendrite = bittensor.dendrite ( config = config, wallet = wallet)

if config.use_wandb: 
    bittensor.wandb(config = config)


##########################
##### Setup Model ######
##########################
model = Nucleus( config = config, wallet = wallet, graph = graph )
model = model.to( config.nucleus.device )
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = config.learning_rate,
    momentum = config.momentum,
)

##########################
##### Load batches ######
##########################
next(dataset)

##########################
##### Run experiment #####
##########################
# Measure state before.
start_time = time.time()
io_1 = psutil.net_io_counters()
start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv
success_results = []
scores_history = []
avg_loss_history = []
table_data = [] 
def step(uids):
    inputs = next(dataset)
    stats, success, scores, losses = model( uids, inputs, dendrite )
    if stats['loss/routing'] == stats['loss/routing'] and stats['loss/routing'] < 8: #true if not nan 
        stats['loss/routing'].backward()
        print('backward!')
    success_results.append(success)
    scores_history.append(scores.detach())
    return stats, success



avg_loss_history = []

target_uids = graph.I.sort(descending = True)[1][torch.tensor(range(0, 4000, int(4000/config.nucleus.sim_network_size)))]
target_uids = torch.cat([target_uids, torch.tensor([3566])])
uids = torch.tensor([])
step_count = 0

while True:
    if len(uids) == 0:
        uids = target_uids[torch.randperm(len(target_uids))]

    stats, success = step( uids[:config.nucleus.n_queried] )
    uids = uids[config.nucleus.n_queried:]
        
    # Apply step.
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    # for k, v in step_stats.items(): 
    #     if k not in stats.keys():
    #         stats[k] = []
    #     if v == v:
    #         stats[k].append(v)
    
    # for k, v in stats.items():
    #     if len(v) > 0:
    #         stats[k] = sum(v)/len(v)
    #         if hasattr(stats[k], 'item'):
    #             stats[k] = stats[k].item()
    
    if stats['loss/routing'] == stats['loss/routing'] and stats['loss/routing'] < 8:
        avg_loss_history.append( stats['loss/routing'] )
    else:
        avg_loss_history.append( 0 )
    
    print ('step:', step_count, '\tavg loss:', avg_loss_history[-1] )

    # average scores
    average_scores = sum( scores_history ) / len(scores_history)
    topk_vals, topk_uids = average_scores.topk( config.nucleus.n_queried )

    if config.use_wandb:
        scores_log = {}
        for k, v in enumerate(average_scores):
            if k in target_uids:
                scores_log['score/target/' + str(k)] = v.item()
            else:
                scores_log['score/' + str(k)] = v.item()

        stats['stat/success'] = sum(success) / len(success)
        stats['stat/num_success'] = sum(success) 
        print(stats)
        wandb.log( {**stats, **scores_log}, step=step_count)
        step_count += 1

# Measure state after.
io_2 = psutil.net_io_counters()
total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv
end_time = time.time()


##########################
##### Show results #######
##########################

total_seconds =  end_time - start_time

total_success = sum([sum(ri) for ri in success_results])
total_sent = config.nucleus.n_queried * config.n_steps
total_failed = total_sent - total_success


print ('\nElapsed:', total_seconds) 
print ('\nSteps:', config.n_steps ) 
print ('Step speed:', config.n_steps / (total_seconds), "/s" ) 
print ('\nQueried:', total_sent )
print ('Query speed:', total_sent / (total_seconds), "/s" ) 
print ('\nBatch size:', config.dataset.batch_size ) 
print ('Sequence Length:', config.dataset.block_size )

print ('\nSuccess', total_success) 
print ('Failed', total_failed ) 
print ('Rate', total_success / (total_success + total_failed))

print ("\nAvg batches per endpoint:", (total_sent / 4096 ))
print ("Avg examples per endpoint:", (total_sent * config.dataset.batch_size / 4096 ))
print ("Avg tokens per endpoint:", ( (total_sent * config.dataset.batch_size * config.dataset.block_size) / 4096 ))
print("\nTotal Upload:", get_size( total_bytes_sent ))
print("Total Download:", get_size( total_bytes_recved ))
print("\nUpload Speed:", get_size( total_bytes_sent/ total_seconds), "/s")
print("Download Speed:", get_size( total_bytes_recved/ total_seconds), "/s")

import sys
sys.exit()