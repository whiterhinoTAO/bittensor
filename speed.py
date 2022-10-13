##################
##### Import #####
##################
import torch
import concurrent.futures
import time
import psutil
import random
import argparse
from tqdm import tqdm
import bittensor


##########################
##### Get args ###########
##########################
parser = argparse.ArgumentParser( 
    description=f"Bittensor Speed Test ",
    usage="python3 speed.py <command args>",
    add_help=True
)
parser.add_argument(
    '--max_workers', 
    dest='max_workers', 
    type=int,
    default=10,
    help='''Maximum concurrent workers on threadpool'''
)
parser.add_argument(
    '--timeout', 
    dest='timeout', 
    type=int,
    default=4,
    help='''Timeout for each set of queries (we always wait this long)'''
)
parser.add_argument(
    "--n_queried", 
    dest='n_queried', 
    type=int,
    default=10,
    help='''The number of endpoints we query each step.'''
)
parser.add_argument(
    "--n_steps", 
    dest='n_steps', 
    type=int,
    default=10,
    help='''The number of concurrent steps we run.'''
)
parser.add_argument(
    "--batch_size", 
    dest='batch_size', 
    type=int,
    default=10,
    help='''Input batch size'''
)
parser.add_argument(
    "--sequence_length", 
    dest='sequence_length', 
    type=int,
    default=20,
    help='''Input sequence length'''
)
bittensor.wallet.add_args(parser)
bittensor.logging.add_args(parser)
bittensor.subtensor.add_args(parser)
config = bittensor.config(parser = parser)

##########################
##### Setup objects ######
##########################
# Sync graph and load power wallet.
bittensor.logging( config = config )
subtensor = bittensor.subtensor( config = config )
graph = bittensor.metagraph( subtensor = subtensor ).sync()
wallet = bittensor.wallet( config = config )

################################
##### Experiment arguments #####
################################
# A list of pre-instantiated endpoints with stub connections.
endpoints = [bittensor.receptor( wallet = wallet, endpoint = graph.endpoint_objs[i] ) for i in range(graph.n)]

# Timeout for each set of queries (we always wait this long)
timeout = config.timeout

# The number of endpoints we query each step.
n_queried = config.n_queried

# The number of concurrent steps we run.
n_steps = config.n_steps

# The number of workers in the thread pool
max_workers = config.max_workers

# The tensor we are going to send over the wire
inputs = torch.ones([config.batch_size, config.sequence_length], dtype=torch.int64)

############################
##### Forward Function #####
############################
# Forward function queries (n_queried) random endpoints with the inputs
# then waits timeout before checking for success from each query.
# The function returns a list of booleans True or false depending on the query result.
def forward():
    
    # Preable.
    futures = []
    result = [False for _ in range(n_queried)]
    
    # Randomly select (n_queried) peers to sample.
    for i in random.sample( range(4096), n_queried ):
        
        # Build the request.
        synapse = bittensor.synapse.TextCausalLMNext()    
        grpc_request = bittensor.proto.TensorMessage (
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            tensors = [ synapse.serialize_forward_request_tensor ( inputs )],
            synapses = [ synapse.serialize_to_wire_proto() ],
            requires_grad = False,
        )
        
        # Fire off the future.
        futures.append( endpoints[i].stub.Forward.future(
            request = grpc_request, 
            timeout = timeout,
            metadata = (
                ('rpc-auth-header','Bittensor'),
                ('bittensor-signature', endpoints[i].sign() ),
                ('bittensor-version',str(bittensor.__version_as_int__)),
                ('request_type', str(bittensor.proto.RequestType.FORWARD)),
            )
          )
        )
        
    # We force the wait period. 
    time.sleep(timeout)
    
    # Iterate over the futures and check for success.
    for i,f in enumerate( futures ):
        try:
            if f.done():
                fresult = f.result()
                if fresult.return_code == 1:
                    # We return a True on success and skip deserialization.
                    #result[i] = synapse.deserialize_forward_response_proto ( inputs, fresult.tensors[0] ).shape
                    result[i] = True
        except:
            pass
        
    # Return the list of booleans.
    return result


##########################
##### Run experiment #####
##########################
# Measure state before.
start_time = time.time()
io_1 = psutil.net_io_counters()
start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv


# Run each query concurrently then get results as completed.
exp_results = []
exp_futures = []
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i in tqdm(range(n_steps), desc='Submitting', leave=True):
        exp_futures.append(executor.submit(forward))
    for future in tqdm(concurrent.futures.as_completed(exp_futures), desc='Filling', leave=True):
        exp_results.append( future.result() )     
        
# Measure state after.
io_2 = psutil.net_io_counters()
total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv
end_time = time.time()

########################
##### Show results #####
########################
def get_size(bytes):
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}B"
        bytes /= 1024
        

total_success = sum([sum(ri) for ri in exp_results])
total_sent = n_queried * n_steps
total_failed = total_sent - total_success

print ('\n')
for arg in vars(args):
    print (arg, ":", getattr(args, arg))

total_seconds =  end_time - start_time
print ('\nElapsed:', total_seconds) 
print ('\nSteps:', n_steps ) 
print ('Step speed:', n_steps / (total_seconds), "/s" ) 

print ('\nQueried:', n_queried )
print ('Query speed:', total_sent / (total_seconds), "/s" ) 

print ('\nSuccess', total_success) 
print ('Failed', total_failed ) 
print ('Rate', total_success / (total_success + total_failed))

print("\nTotal Upload:", get_size( total_bytes_sent ))
print("Total Download:", get_size( total_bytes_recved ))
print("\nUpload Speed:", get_size( total_bytes_sent/ total_seconds), "/s")
print("Download Speed:", get_size( total_bytes_recved/ total_seconds), "/s")



