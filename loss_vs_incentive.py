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
import math 

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
        self.graph = graph
        self.sigmoid = torch.nn.Sigmoid()
        self.synapse = bittensor.TextCausalLMNext()
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.tokenizer = bittensor.tokenizer()
        self.pad_token = self.tokenizer(self.tokenizer.pad_token)['input_ids'][0]
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.receptors = [bittensor.receptor( wallet=self.wallet, endpoint = self.graph.endpoint_objs[i] ) for i in range(self.graph.n)]
        self.model_baseline = server(config = config, model_name = 'EleutherAI/gpt-neo-125M')

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
        self.decoder = TransformerEncoder( 
            TransformerEncoderLayer( 
                bittensor.__network_dim__, 
                config.nucleus.nhead, 
                config.nucleus.nhid, 
                config.nucleus.dropout, 
                batch_first=True
            ), 
            config.nucleus.nlayers 
        )
        self.decoder_head = torch.nn.Linear( 
            bittensor.__network_dim__, 
            bittensor.__vocab_size__ , 
            bias=False
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

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        return bittensor.config( parser )

    def cal_loss(self, inputs, query_response, validation_len = 8, p = False):
        inputs_nxt = inputs[..., -validation_len:]
        _losses_val, _losses = phrase_cross_entropy(inputs_nxt, query_response, reduce=True, p = p)
        return _losses

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
        random_endpoints = [graph.endpoints[uid] for uid in uids]

        with torch.no_grad():
            start_time = time.time()
            responses, return_ops, times = dendrite.text(
                    endpoints=random_endpoints,
                    inputs=inputs_seq,
                    synapses=[self.synapse],
                    timeout=self.config.nucleus.timeout
                )

            dend_forward_time = time.time()
            qps = sum([ops == 1 for ops in return_ops]) / (time.time() - start_time)
            response_baseline = self.model_baseline.encode_forward_causallmnext(inputs)[1].logits
        
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
            return stats, successes, routing_score, {}
            
        for i, (r, op) in enumerate(list(zip(responses, return_ops))):
            if op == 1 and r[0][:, :, 0].sum() < 0:
                return_ops[i] = 0

        response_success = [r for r, op in list(zip(responses, return_ops)) if op == 1]
        time_success = [t for t, op in list(zip(times, return_ops)) if op == 1]

        # Compute loss.
        losses = {}
        for uid, r, op in list(zip(uids, responses, return_ops)):
            if op == 1:
                losses[uid] = self.cal_loss(inputs, r[0][:, :, :2], self.config.nucleus.validation_len)
        # print (losses)
        loss_min = min(list(losses.values()))

        stats = {
            'loss/min': loss_min,
            'loss/count': len(losses),
            'loss/average': sum(losses) / len(losses),
            'stat/receptor_time_avg': sum(time_success) / sum(successes),
            'stat/dend_time': dend_forward_time - start_time,
            'stat/batch_time': time.time() - start_time,
            'stat/qps': qps,
        }
        print(stats)
        
        # Return.
        return stats, successes, routing_score, losses
    
    
##############################
##### Build config ###########
##############################
parser = argparse.ArgumentParser( 
    description=f"Bittensor Validator Training ",
    usage="python3 train.py <command args>",
    add_help=True
)
parser.add_argument( '--max_workers', type=int, default=10, help='''Maximum concurrent workers on threadpool''')
parser.add_argument( '--n_steps', type=int, default=10, help='''The number of steps we run.''')
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
subtensor = bittensor.subtensor( config = config )
graph = bittensor.metagraph( subtensor = subtensor ).sync()
wallet = bittensor.wallet( config = config )
dendrite = bittensor.dendrite ( config = config, wallet = wallet)



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
neuron_losses = {k: [] for k in range(graph.n)} 
last_losses = {}
def step(idx, uids):
    inputs = next(dataset)
    stats, success, scores, losses = model( uids, inputs, dendrite )
    success_results.append(success)
    scores_history.append(scores.detach())
    for k,v in losses.items():
        neuron_losses[k].append(v)
    return stats, success, losses


if config.use_wandb: 
    bittensor.wandb(config = config)

avg_loss_history = []
perm_uids = list(range(graph.n))
with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
    step_chunks = list( chunks( list(range(config.n_steps * math.ceil( graph.n /config.nucleus.n_queried) )), config.chunk_size ) )
    for ci, chunk in enumerate( step_chunks ):
        
        # Fire batches.
        chunk_futures = []
        chunk_results = []
        for i in chunk:
            if len(perm_uids) == 0:
                perm_uids = list(range(graph.n))
            target_uids = perm_uids[:config.nucleus.n_queried]
            chunk_futures.append(executor.submit(step, i, target_uids))
            del perm_uids[:config.nucleus.n_queried]

        for future in concurrent.futures.as_completed(chunk_futures):
            chunk_results.append( future.result() )
            
        # Apply step.
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        stats = {}
        # Aggregating stats
        for r in chunk_results:
            for k, v in r[0].items():
                if k not in stats.keys():
                    stats[k] = []
                if v == v:
                    stats[k].append(v)
        
        losses = {}
        # Aggregate losses
        for r in chunk_results:
            for k, v in r[2].items():
                if k not in stats.keys():
                    losses['losses/' + str(k)] = []
                if v == v:
                    losses['losses/' + str(k)].append(v)
        
        for k, v in stats.items():
            if len(v) > 0:
                stats[k] = sum(v)/len(v)
                if hasattr(stats[k], 'item'):
                    stats[k] = stats[k].item()
        
        for k, v in losses.items():
            if len(v) > 0:
                losses[k] = sum(v)/len(v)
                if hasattr(losses[k], 'item'):
                    losses[k] = losses[k].item()
        
        successes = [ sum(l[1]) / len(l[1]) for l in chunk_results]
        # avg_loss_history.append( stats['loss/routing'] )
        avg_loss_history.append( 0 )
        print ('step:', ci+1, '/', len(step_chunks), '\tavg loss:', avg_loss_history[-1] )

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

            stats['stat/success'] = sum(successes) / len(successes)
            stats['stat/num_success'] = sum(successes) 
            wandb.log( {**stats, **scores_log, **losses}, step=ci)
    
        if ci % math.ceil( graph.n /config.nucleus.n_queried) * 5 == 0:
            neuron_losses_mean = []
            for uid, losses in neuron_losses.items():
                if len(losses) > 0:
                    neuron_losses_mean.append( [uid, graph.I[uid].item(), (sum(losses)/len(losses)).item(), len(losses)] )

            table = wandb.Table(data=neuron_losses_mean, columns = ["uid", "incentive", "loss", "count"])
            wandb.log({"incentive_vs_loss" : wandb.plot.scatter(table, "incentive", "loss", title="incentive VS loss")})

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