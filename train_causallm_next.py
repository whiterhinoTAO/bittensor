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
from tqdm import tqdm
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from bittensor.utils.tokenizer_utils import phrase_cross_entropy
import wandb
from pympler import summary, muppy

from rich.traceback import install
install(show_locals=False)

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
class PositionalEncoding(nn.Module):
    r""" Positional Encoder which adds information based on the relative position of each token

    """
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # === Create position matrix ===
        # Creates a positional matrix with alternating frequencies
        # pe: (torch.FloatTensor) positional encoding matrix
        # pe.shape: [1, max_len, network_dim]
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # === Positional Encoding ===
        # Inject some information of the relative position of the token in the sequence.
        #  Finally, Dropout is applied to tokens
        # x: (torch.FloatTensor) input sequence tokens with position information injected
        # x.shape: [batch_size, seq_len, network_dim]
        x = x + self.pe[0, :x.size(1)]
        return self.dropout(x)
    
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
        self.receptors = [bittensor.receptor( wallet=self.wallet, endpoint = self.graph.endpoint_objs[i] ) for i in range(self.graph.n)]

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
        parser.add_argument('--nucleus.n_queried', type=int, default=50, help='''The number of endpoints we query each step.''')
        parser.add_argument('--nucleus.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        return bittensor.config( parser )
        
    def query( self, uids, inputs ):
        futures = []
        results = [self.synapse.nill_forward_response_tensor(inputs) for _ in uids ]
        for index, uid in enumerate(uids):        
            grpc_request = bittensor.proto.TensorMessage (
                version = bittensor.__version_as_int__,
                hotkey = self.wallet.hotkey.ss58_address,
                tensors = [ self.synapse.serialize_forward_request_tensor ( inputs )],
                synapses = [ self.synapse.serialize_to_wire_proto() ],
                requires_grad = False,
            )
            futures.append( self.receptors[uid].stub.Forward.future(
                request = grpc_request, 
                timeout = self.config.nucleus.timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature', self.receptors[uid].sign() ),
                    ('bittensor-version',str(bittensor.__version_as_int__)),
                    ('request_type', str(bittensor.proto.RequestType.FORWARD)),
                )
              )
            )
        time.sleep( self.config.nucleus.timeout )
        for i,f in enumerate( futures ):
            try:
                if f.done():
                    fresult = f.result()
                    if fresult.return_code == 1:
                        response_tensor = self.synapse.deserialize_forward_response_proto ( inputs, fresult.tensors[0] )
                        results[index] = response_tensor

            except Exception as e:
                # Unknown error logging.
                pass

        return [ r.to(self.config.nucleus.device) for r in results ] 
    
    def base_params(self, query_response, inputs_nxt):
        # topk_tensor = unravel_topk_token_phrases(query_response, topk=synapse.topk)  # [batch_size, topk + 1, max_len]
        _losses_val, _losses = phrase_cross_entropy(inputs_nxt, query_response, reduce=False)
        _losses_val[_losses_val.isnan()] = 20  # assign large loss
        _losses[_losses.isnan()] = 20  # assign large loss
        _loss_val = _losses_val.mean()
        _loss = _losses.mean()
        
        return _loss, _loss_val, _losses

    def forward(self, inputs, dendrite):
        inputs = inputs.to(self.config.nucleus.device)
        inputs_seq = inputs[..., :-self.config.validation_len]
        inputs_nxt = inputs[..., -self.config.validation_len:] 

        # Route
        embedding = self.token_embedding(inputs_seq) * math.sqrt(bittensor.__network_dim__)
        src_mask = torch.triu(torch.ones(embedding.size(1), embedding.size(1)) * float('-inf'), diagonal=1).to(self.config.nucleus.device)
        pos_embedding = self.local_pos_encoder(embedding)
        routing_context = self.encoder(pos_embedding, mask=src_mask)
        routing_score = torch.mean(self.sigmoid(self.gates(routing_context[:, -1, :])), dim=0)
        
        # Query
        uid_sample = random.sample( range(4096), self.config.nucleus.n_queried )
        random_endpoints = [graph.endpoints[uid] for uid in uid_sample]
        with torch.no_grad():
            #responses = self.query( uid_sample, inputs_seq)
            responses, return_ops, times = dendrite.text(
                endpoints=random_endpoints,
                inputs=inputs_seq,
                synapses=[self.synapse],
                timeout=self.config.nucleus.timeout
            )
        with torch.no_grad():
            losses = []
            for response in responses:
                loss , _ , base_losses  = self.base_params(response[0], inputs_nxt)
                losses += [base_losses]

        norm_routing_score =  routing_score[uid_sample] /  routing_score[uid_sample].sum()
        # Evaluate.
        weighted_probs = sum([ torch.exp(-r)  * w for r, w in list(zip( losses, norm_routing_score)) ])
        loss = -torch.log(weighted_probs).mean()
        return loss
    
    
##############################
##### Build config ###########
##############################
parser = argparse.ArgumentParser( 
    description=f"Bittensor Speed Test ",
    usage="python3 speed.py <command args>",
    add_help=True
)
parser.add_argument( '--max_workers', type=int, default=10, help='''Maximum concurrent workers on threadpool''')
parser.add_argument( '--n_steps', type=int, default=10, help='''The number of steps we run.''')
parser.add_argument( '--chunk_size', type=int, default=10, help='''The number of concurrent steps we run.''')
parser.add_argument( '--learning_rate', type=float, help='Training initial learning rate.', default=0.01)
parser.add_argument( '--momentum', type=float, help='optimizer momentum.', default=0.8)
parser.add_argument( '--validation_len', type=int, default=5, help='''validation length of the sequence''')
parser.add_argument( '--use_wandb', action='store_true', default=False, help='''To use wandb to track results''')

bittensor.wallet.add_args(parser)
bittensor.logging.add_args(parser)
bittensor.subtensor.add_args(parser)
bittensor.dataset.add_args(parser)
bittensor.wandb.add_args(parser)
bittensor.dendrite.add_args(parser)
Nucleus.add_args(parser)
config = bittensor.config(parser = parser)
print (config)

##########################
##### Setup objects ######
##########################
# Sync graph and load power wallet.
bittensor.logging( config = config )
subtensor = bittensor.subtensor( config = config )
dataset = bittensor.dataset( config = config , num_batches = config.n_steps+1 , block_size=subtensor.validator_sequence_length + config.validation_len)
graph = bittensor.metagraph( subtensor = subtensor ).sync()
wallet = bittensor.wallet()
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
import queue
dataqueue = queue.Queue()
for i in tqdm( range(config.n_steps + 1 ), desc='Loading dataset...', leave=True):
    dataqueue.put( next(dataset) )

##########################
##### Run experiment #####
##########################
# Measure state before.
start_time = time.time()
io_1 = psutil.net_io_counters()
start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv

def step():
    inputs = dataqueue.get().to(config.nucleus.device)
    loss = model( inputs , dendrite)
    
    return loss

avg_loss_history = []
if config.use_wandb: 
    bittensor.wandb(config= config)


threadpool = concurrent.futures.ThreadPoolExecutor( max_workers = config.max_workers )
step_chunks = list( chunks( list(range(config.n_steps)), config.chunk_size ) )
for ci, chunk in enumerate( step_chunks ):
    # Clear grads.
    optimizer.zero_grad() 
    
    # Fire batches.
    chunk_futures = []
    chunk_results = []
    for i in chunk:
        chunk_futures.append(threadpool.submit(step))
        
    for future in concurrent.futures.as_completed(chunk_futures):
        chunk_results.append( future.result() )  
            
    # Apply step.
    losses = sum([l for l in chunk_results])

    losses.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    avg_loss_history.append( losses.detach()/ config.chunk_size )
    if config.use_wandb:
        wandb.log({'loss':  losses.detach()/ config.chunk_size}, step=ci)
    print ('step:', ci+1, '/', len(step_chunks), '\tavg loss:', avg_loss_history[-1] )


# Measure state after.
io_2 = psutil.net_io_counters()
total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv
end_time = time.time()


##########################
##### Show results #######
##########################

total_seconds =  end_time - start_time
total_sent = config.nucleus.n_queried * config.n_steps

print ('\nElapsed:', total_seconds) 
print ('\nSteps:', config.n_steps ) 
print ('Step speed:', config.n_steps / (total_seconds), "/s" ) 
print ('\nQueried:', total_sent )
print ('Query speed:', total_sent / (total_seconds), "/s" ) 
print ('\nBatch size:', config.dataset.batch_size ) 
print ('Sequence Length:', config.dataset.block_size ) 
print ("\nAvg batches per endpoint:", (total_sent / 4096 ))
print ("Avg examples per endpoint:", (total_sent * config.dataset.batch_size / 4096 ))
print ("Avg tokens per endpoint:", ( (total_sent * config.dataset.batch_size * config.dataset.block_size) / 4096 ))
print("\nTotal Upload:", get_size( total_bytes_sent ))
print("Total Download:", get_size( total_bytes_recved ))
print("\nUpload Speed:", get_size( total_bytes_sent/ total_seconds), "/s")
print("Download Speed:", get_size( total_bytes_recved/ total_seconds), "/s")

import sys
sys.exit()