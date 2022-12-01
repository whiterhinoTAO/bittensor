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
        self.graph = graph
        self.sigmoid = torch.nn.Sigmoid()
        self.synapse = bittensor.TextCausalLMNext()
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.tokenizer = bittensor.tokenizer()
        self.pad_token = self.tokenizer(self.tokenizer.pad_token)['input_ids'][0]
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.receptors = [bittensor.receptor( wallet=self.wallet, endpoint = self.graph.endpoint_objs[i] ) for i in range(self.graph.n)]
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
        return _losses_val

    def mix_response(self, response_success, normalized_topk_routing_scores):
        batch_size = response_success[0][0].shape[0]
        topk =  response_success[0][0].shape[1] - 1
        mixed_response = torch.zeros(batch_size, bittensor.__vocab_size__ + 1  , 2)
        all_logits = torch.tensor(list(range(bittensor.__vocab_size__)))
        mixed_response[:, : -1, 1] = all_logits.repeat(batch_size, 1)

        for r, w in list(zip(response_success, normalized_topk_routing_scores)):
            response = torch.zeros(batch_size, bittensor.__vocab_size__ , 2)
            response[:, :, 1] = all_logits.repeat(batch_size, 1)

            for batch in range(batch_size):
                r_batch = r[0][batch, : -1, :]
                r_batch_sorted = r_batch[r_batch[:, 0].sort(descending = False)[1]]
                index = r_batch_sorted[:, 1].long()
                prob = r_batch_sorted[:, 0] 
                response[batch, index, 0] = prob
            mixed_response[:, :-1, 0] += w * response[:, :, 0]

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
        random_endpoints = [graph.endpoints[uid] for uid in uids]
        topk_routing_scores = routing_score[uids]

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
            return stats, successes, routing_score
            
        for i, (r, op) in enumerate(list(zip(responses, return_ops))):
            if op == 1 and r[0][:, :, 0].sum() < 0:
                return_ops[i] = 0

        responses_gptj = responses[-1:]
        return_ops_gptj = return_ops[-1:]
        times_gptj = times[-1:]
        
        responses = responses[:-1]
        return_ops = return_ops[:-1]
        times = times[:-1]

        uid_success = [uid for uid, op in list(zip(uids, return_ops)) if op == 1]
        response_success = [r for r, op in list(zip(responses, return_ops)) if op == 1]
        time_success = [t for t, op in list(zip(times, return_ops)) if op == 1]
        normalized_topk_routing_scores = topk_routing_scores[successes]/topk_routing_scores[successes].sum()
        mixed_response = self.mix_response(response_success, normalized_topk_routing_scores)
        # mixed_ind_response = [self.mix_response([r], torch.tensor([1])) for r in response_success]
        averaged_response = self.mix_response(response_success, torch.ones_like(normalized_topk_routing_scores) / len(normalized_topk_routing_scores))
        # Compute loss.
        loss_routing = self.cal_loss(inputs, mixed_response, self.config.nucleus.validation_len)
        loss_routing_baseline = self.cal_loss(inputs, averaged_response, self.config.nucleus.validation_len)
        losses = torch.tensor([self.cal_loss(inputs, r[0], self.config.nucleus.validation_len) for r in response_success])
        top_mix_loss = {}
        for i in range(1, min(11, len(response_success)), 2):        
            top_mix_response = self.mix_response( [response_success[idx] for idx in losses.sort()[1][:i]], torch.ones(i) / i)
            top_uid = [uid_success[idx] for idx in losses.sort()[1][:i]]
            print(f'===== top_mix{i} ======\n', top_uid)
            loss_top_mix = self.cal_loss(inputs, top_mix_response, self.config.nucleus.validation_len, p = False)
            print('loss: ', loss_top_mix)
            top_mix_loss[f'loss/top_mix_{i}'] = loss_top_mix
            
        
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
        if return_ops_gptj[0] == 1:
            stats['loss/gptj'] = self.cal_loss(inputs, responses_gptj[0][0], self.config.nucleus.validation_len)
        
        stats = {**stats, **top_mix_loss}
        stats = {**stats}
        # print(stats)
        
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
table_data = [] 
def step(idx, uids):
    inputs = next(dataset)
    stats, success, scores, losses = model( uids, inputs, dendrite )
    if stats['loss/routing'] == stats['loss/routing'] and stats['loss/routing'] < 8: #true if not nan 
        stats['loss/routing'].backward()
        print('backward!')
    success_results.append(success)
    scores_history.append(scores.detach())
    return stats, success


if config.use_wandb: 
    bittensor.wandb(config = config)

avg_loss_history = []
# target_uids = graph.I.sort(descending = True)[1][:config.nucleus.n_queried]
# top 500
# target_uids = torch.tensor ([
#     3546, 3913, 2720, 288, 2716, 3857, 448, 3673, 3948, 3821, 3924, 3645, 3794, 2757, 2930, 3967, 3609, 3702, 3774, 3951, 3687, 3690, 3870, 3735, 3551, 3932, 286, 3692, 3770, 3940, 3566, 3710, 3795, 3705, 3750, 3955, 3723, 3585, 3665, 3753, 3976, 3950, 3530, 3874, 3738, 3670, 3882, 3630, 3600, 3668, 3536, 2277, 3651, 3871, 3984, 3972, 3939, 277, 3529, 358, 416, 385, 452, 287, 2955, 306, 376, 2476, 103, 315, 3985, 3595, 222, 3988, 203, 54, 490, 3803, 28, 152, 3621, 3649, 3779, 3822, 3880, 3904, 100, 3, 3925, 476, 50, 136, 3519, 3745, 3558, 3845, 3544, 3633, 3749, 3793, 339, 40, 3970, 2854, 3576, 3571, 3646, 3829, 149, 116, 354, 3943, 3689, 3695, 3899, 252, 3583, 3947, 3531, 3656, 372, 2524, 3737, 2904, 3503, 2938, 3553, 3619, 3941, 3961, 2876, 2687, 471, 2878, 2893, 2703, 2859, 460, 3811, 3847, 2690, 3775, 2776, 2533, 2544, 2846, 3722, 3568, 2853, 2578, 2655, 2669, 2787, 2914, 2646, 2569, 2592, 2644, 2969, 391, 2514, 2880, 2869, 4012, 2964, 485, 55, 2932, 209, 2552, 2760, 4029, 4044, 4054, 4055, 4053, 320, 237, 336, 402, 431, 478, 4049, 31, 43, 200, 261, 3708, 419, 4063, 183, 249, 3863, 233, 292, 470, 352, 3524, 267, 145, 427, 424, 107, 2953, 270, 9, 259, 3534, 243, 335, 380, 409, 3796, 468, 3730, 169, 245, 202, 47, 246, 3556, 213, 425, 1368, 3654, 16, 3746, 77, 82, 146, 3613, 477, 1, 39, 357, 48, 3995, 3797, 404, 3521, 109, 3962, 137, 42, 4003, 3715, 4030, 4045, 4089, 3592, 3833, 18, 2815, 3652, 2535, 96, 323, 4074, 4088, 255, 4086, 151, 3526, 1488, 3540, 3562, 3676, 3873, 3928, 3964, 3989, 3960, 3979, 4093, 3931, 3929, 3664, 710, 2931, 610, 329, 4047, 3608, 449, 2863, 346, 3703, 3622, 197, 218, 225, 389, 398, 454, 483, 2181, 3716, 2574, 497, 1487, 4014, 4020, 3545, 2411, 210, 4022, 4037, 10, 3505, 3515, 3523, 3574, 3581, 3663, 3755, 3761, 3843, 3987, 58, 171, 3956, 2917, 2618, 2820, 2987, 3854, 2894, 3629, 3552, 3518, 3563, 2528, 2689, 2833, 2868, 2934, 2945, 2954, 4090, 46, 97, 2935, 1553, 2712, 401, 3841, 2600, 2555, 3893, 3587, 262, 2183, 447, 3700, 3968, 3658, 2590, 2319, 2128, 3577, 2448, 410, 422, 3628, 3707, 3759, 3804, 2160, 2458, 457, 75, 3982, 3611, 3762, 3998, 248, 465, 3659, 2662, 236, 2678, 2809, 3579, 2530, 256, 399, 327, 3839, 353, 21, 83, 158, 2511, 2559, 2679, 2409, 3999, 3983, 2247, 365, 3648, 3496, 2056, 2255, 143, 2417, 2381, 122, 3756, 1408, 3599, 2091, 2027, 3885, 2354, 2266, 3927, 2132, 3573, 2192, 361, 3550, 3565, 3824, 3851, 2112, 2, 2114, 2149, 2225, 2379, 2300, 2406, 2011, 135, 2005, 2002, 2463, 3922, 3846, 2759, 2032, 3760, 3806, 2422, 1429, 2413, 2208, 3949, 3912, 2705, 2361, 2975, 32, 2968, 2976, 2012, 1477, 2304, 2507, 2585, 2614, 2653, 2949, 2735, 2000, 2059, 2761, 2563, 2832, 2176, 2352, 1999, 2069, 2209, 2307, 2161, 3783, 3729, 2632, 3643, 3831, 1213, 3709, 3666, 2324, 2495, 2755, 2503, 2505, 2602, 2622, 2638, 2728, 2769, 2852, 2908, 3527, 3525, 3744, 2736
# ])

# top 500 -> 300 unique
# target_uids = torch.tensor([
#     3546, 3913, 2720, 448, 3673, 3948, 2757, 3774, 3951, 286, 3566, 3670, 3651, 3595, 3779, 3, 3558, 3633, 3749, 3576, 3943, 3531, 3941, 2703, 2859, 3775, 3568, 2787, 2552, 4055, 320, 402, 183, 107, 2953, 243, 3730, 246, 425, 77, 477, 404, 96, 4088, 255
# ])
# top 100
# target_uids = torch.tensor([
#     3546, 3913, 2498, 2720, 2146, 2831, 2204, 288, 1160, 2716, 3857, 448, 253, 3673, 3948, 3821, 3924, 3645, 3794, 3488, 
#     2757, 2930, 908, 3967, 2070, 3609, 3702, 3774, 3951, 3687, 2872, 2939, 2977, 3690, 3870, 3735, 3551, 3932, 286, 3692, 
#     3770, 3940, 3566, 3710, 3795, 3705, 3750, 3955, 3723, 3585, 3665, 3753, 3976, 3950, 3530, 3874, 3738, 3670, 3882, 3630, 
#     3600, 3668, 2770, 2796, 3536, 2277, 3651, 3871, 3984, 3972, 3939, 277, 3529, 358, 416, 385, 3209, 452, 287, 2955, 306, 376, 
#     3016, 2476, 103, 1481, 315, 3985, 1202, 1218, 3595, 1320, 1377, 222, 3988, 203, 54, 3111, 490, 3803, 28
# ])

# top 100 - 2
# target_uids = torch.tensor([
#     3625, 3994, 3555, 3891, 3717, 3260, 3819, 3660, 3782, 3632, 3739, 3624, 3639, 3932, 3688, 3771, 3670, 3855, 
#     3911, 3514, 3584, 3860, 3926, 3965, 3909, 3605, 3616, 3996, 3930, 3816, 3951, 3767, 3591, 3821, 3862, 3757, 
#     3666, 3967, 3653, 3501, 82, 243, 380, 409, 146, 3519, 3722, 335, 151, 3742, 2963, 323, 4074, 489, 355, 23, 
#     170, 2498, 3619, 221, 62, 211, 289, 5, 297, 370, 110, 94, 450, 166, 262, 11, 308, 284, 2520, 3961, 2555, 2664, 
#     2708, 3500, 2954, 2542, 2607, 2688, 2630, 2666, 2689, 2868, 2945, 2899, 2597, 2712, 2942, 2685, 2804, 3528, 2547, 3799, 2918, 2609
# ])

# unique from top 100 - 2
# target_uids = torch.tensor([3625, 3260, 3932, 3670, 3666, 3722, 151, 323, 170, 284, 2708])
# target_uids = torch.tensor([ 3625, 

df = pd.read_csv('loss_vs_incentive2.csv')
target_uids = df[df['count'] > 200].sort_values('loss')['uid'][:1000].values
uids = []
with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
    step_chunks = list( chunks( list(range(config.n_steps)), config.chunk_size ) )
    for ci, chunk in enumerate( step_chunks ):
        
        # Fire batches.
        chunk_futures = []
        chunk_results = []
        for i in chunk:
            if len(uids) == 0:
                uids = target_uids         
            chunk_futures.append(executor.submit(step, i, torch.concat([uids[:config.nucleus.n_queried], torch.tensor(4049)])))
            uids = uids[config.nucleus.n_queried:]
            # chunk_futures.append(executor.submit(step, i, target_uids))
        
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
        
        for k, v in stats.items():
            if len(v) > 0:
                stats[k] = sum(v)/len(v)
                if hasattr(stats[k], 'item'):
                    stats[k] = stats[k].item()
        
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
            print(stats)
            wandb.log( {**stats, **scores_log}, step=ci)

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