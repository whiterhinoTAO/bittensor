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
from bittensor._neuron.text.neuron_utilities import ThreadQueue, PositionalEncoding, calc_loss_fct
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
class Nucleus(nn.Module):
    def __init__(self, config, wallet, graph ):
        super(Nucleus, self).__init__()
        self.config = config
        self.wallet = wallet
        self.graph = graph
        self.sigmoid = torch.nn.Sigmoid()
        self.synapse = bittensor.TextCausalLM()
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
        successes = [ False for _ in uids ]
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
            bittensor.logging.rpc_log ( 
                axon = False, 
                forward = True, 
                is_response = False, 
                code = bittensor.proto.ReturnCode.Success, 
                call_time = 0, 
                pubkey = self.receptors[uid].endpoint.hotkey, 
                uid = self.receptors[uid].endpoint.uid, 
                inputs = list(inputs.shape), 
                outputs = None, 
                message = 'Success',
                synapse = self.synapse.synapse_type
            )
        time.sleep( self.config.nucleus.timeout )
        for i,f in enumerate( futures ):
            try:
                if f.done():
                    fresult = f.result()
                    if fresult.return_code == 1:
                        response_tensor = self.synapse.deserialize_forward_response_proto ( inputs, fresult.tensors[0] )
                        results[index] = response_tensor
                        successes[index] = True
                        bittensor.logging.rpc_log ( 
                            axon = False, 
                            forward = True, 
                            is_response = True, 
                            code = bittensor.proto.ReturnCode.Success, 
                            call_time = self.config.nucleus.timeout, 
                            pubkey = self.receptors[uid].endpoint.hotkey, 
                            uid = self.receptors[uid].endpoint.uid, 
                            inputs = list(inputs.shape), 
                            outputs = list(response_tensor.shape), 
                            message = 'Success',
                            synapse = self.synapse.synapse_type
                        )
                else:
                    # Timeout Logging.
                    bittensor.logging.rpc_log ( 
                        axon = False, 
                        forward = True, 
                        is_response = True, 
                        code = bittensor.proto.ReturnCode.Timeout, 
                        call_time = self.config.nucleus.timeout, 
                        pubkey = self.receptors[uid].endpoint.hotkey, 
                        uid = self.receptors[uid].endpoint.uid, 
                        inputs = list(inputs.shape), 
                        outputs = None, 
                        message = 'Timeout',
                        synapse = self.synapse.synapse_type
                    )
            except Exception as e:
                # Unknown error logging.
                bittensor.logging.rpc_log ( 
                    axon = False, 
                    forward = True, 
                    is_response = True, 
                    code = bittensor.proto.ReturnCode.UnknownException, 
                    call_time = self.config.nucleus.timeout, 
                    pubkey = self.receptors[uid].endpoint.hotkey, 
                    uid = self.receptors[uid].endpoint.uid, 
                    inputs = list(inputs.shape), 
                    outputs = None, 
                    message = str(e),
                    synapse = self.synapse.synapse_type
                )   
                pass
        return [ r.to(self.config.nucleus.device) for r in results ], successes

    def cal_loss(self, inputs, query_response, validation_len = 1):
        
        _logits = query_response.contiguous()
        _labels = inputs.contiguous()
        loss = torch.nn.CrossEntropyLoss()(_logits.view(-1, _logits.size(-1)), _labels.view(-1))
        return loss

    def forward(self, inputs):
        inputs = inputs.to(self.config.nucleus.device)

        # Route
        embedding = self.token_embedding(inputs) * math.sqrt(bittensor.__network_dim__)
        src_mask = torch.triu(torch.ones(embedding.size(1), embedding.size(1)) * float('-inf'), diagonal=1).to(self.config.nucleus.device)
        pos_embedding = self.local_pos_encoder(embedding)
        routing_context = self.encoder(pos_embedding, mask=src_mask)
        routing_score = torch.mean(self.sigmoid(self.gates(routing_context[:, -1, :])), dim=0)
        
        # Query
        uid_sample = random.sample( range(4096), self.config.nucleus.n_queried )
        responses, successes = self.query( uid_sample, inputs)
        
        # Evaluate.
        weighted_responses = sum([ r  * w for r, w in list(zip( responses, routing_score[uid_sample])) ])
        loss = self.cal_loss(inputs, weighted_responses )
        
        return loss, successes
    
    
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

bittensor.wallet.add_args(parser)
bittensor.logging.add_args(parser)
bittensor.subtensor.add_args(parser)
bittensor.dataset.add_args(parser)
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
wallet = bittensor.wallet( name = 'opentensor', hotkey = 'miner-valid-07' )


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
success_results = []

def step():
    inputs = dataqueue.get().to(config.nucleus.device)
    loss, successes = model( inputs )
    loss = loss / config.chunk_size
    loss.backward()
    success_results.append(successes)
    return loss

avg_loss_history = []
with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
    step_chunks = list( chunks( list(range(config.n_steps)), config.chunk_size ) )
    for ci, chunk in enumerate( step_chunks ):
        # Clear grads.
        optimizer.zero_grad() 
        
        # Fire batches.
        chunk_futures = []
        chunk_results = []
        for i in chunk:
            chunk_futures.append(executor.submit(step))
            
        for future in concurrent.futures.as_completed(chunk_futures):
            chunk_results.append( future.result() )  
                
        # Apply step.
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses = [l.item() for l in chunk_results]
        avg_loss_history.append( sum( losses )/ config.chunk_size )
        
        print ('step:', ci+1, '/', len(step_chunks), '\tavg loss:', avg_loss_history[-1] )

        
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