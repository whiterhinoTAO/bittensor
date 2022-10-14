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
    default=100,
    help='''The number of endpoints we query each step.'''
)
parser.add_argument(
    "--n_trials", 
    dest='n_trials', 
    type=int,
    default=3,
    help='''The number of test trials.'''
)
parser.add_argument(
    "--n_tasks", 
    dest='n_tasks', 
    type=int,
    default=1,
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
print (config)

##########################
##### Setup objects ######
##########################
# Sync graph and load power wallet.
bittensor.logging( config = config )
subtensor = bittensor.subtensor( config = config )
graph = bittensor.metagraph( subtensor = subtensor ).sync()
wallet = bittensor.wallet( config = config )
dendrite = bittensor.dendrite(wallet = wallet)

################################
##### Experiment arguments #####
################################
# A list of pre-instantiated endpoints with stub connections.
receptors = [bittensor.receptor( wallet = wallet, endpoint = graph.endpoint_objs[i] ) for i in range(graph.n)]

# Timeout for each set of queries (we always wait this long)
timeout = config.timeout

# The number of endpoints we query each step.
n_queried = config.n_queried

# The number of concurrent steps we run.
n_tasks = config.n_tasks

n_trials = config.n_trials

# The number of workers in the thread pool
max_workers = config.max_workers

# The tensor we are going to send over the wire
inputs = torch.ones([config.batch_size, config.sequence_length], dtype=torch.int64)

synapses = [bittensor.synapse.TextCausalLMNext()]
############################
##### Forward Function #####
############################
# Forward function queries (n_queried) random endpoints with the inputs
# then waits timeout before checking for success from each query.
# The function returns a list of booleans True or false depending on the query result.
def forward(n_queried = n_queried):
    
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
        futures.append( receptors[i].stub.Forward.future(
            request = grpc_request, 
            timeout = timeout,
            metadata = (
                ('rpc-auth-header','Bittensor'),
                ('bittensor-signature', receptors[i].sign() ),
                ('bittensor-version',str(bittensor.__version_as_int__)),
                ('request_type', str(bittensor.proto.RequestType.FORWARD)),
            )
          )
        )

        # Logging
        bittensor.logging.rpc_log ( 
            axon = False, 
            forward = True, 
            is_response = False, 
            code = bittensor.proto.ReturnCode.Success, 
            call_time = 0, 
            pubkey = receptors[i].endpoint.hotkey, 
            uid = receptors[i].endpoint.uid, 
            inputs = list(inputs.shape), 
            outputs = None, 
            message = 'Success',
            synapse = synapse.synapse_type
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
                    response_tensor = synapse.deserialize_forward_response_proto ( inputs, fresult.tensors[0] )
                    result[i] = True

                    # Success logging.
                    bittensor.logging.rpc_log ( 
                        axon = False, 
                        forward = True, 
                        is_response = True, 
                        code = bittensor.proto.ReturnCode.Success, 
                        call_time = timeout, 
                        pubkey = receptors[i].endpoint.hotkey, 
                        uid = receptors[i].endpoint.uid, 
                        inputs = list(inputs.shape), 
                        outputs = list(response_tensor.shape), 
                        message = 'Success',
                        synapse = synapse.synapse_type
                    )

                else:
                    # Timeout Logging.
                    bittensor.logging.rpc_log ( 
                        axon = False, 
                        forward = True, 
                        is_response = True, 
                        code = bittensor.proto.ReturnCode.Timeout, 
                        call_time = timeout, 
                        pubkey = receptors[i].endpoint.hotkey, 
                        uid = receptors[i].endpoint.uid, 
                        inputs = list(inputs.shape), 
                        outputs = None, 
                        message = 'Timeout',
                        synapse = synapse.synapse_type
                    )

        except Exception as e:
            # Unknown error logging.
            bittensor.logging.rpc_log ( 
                axon = False, 
                forward = True, 
                is_response = True, 
                code = bittensor.proto.ReturnCode.UnknownException, 
                call_time = timeout, 
                pubkey = receptors[i].endpoint.hotkey, 
                uid = receptors[i].endpoint.uid, 
                inputs = list(inputs.shape), 
                outputs = None, 
                message = str(e),
                synapse = synapse.synapse_type
            )
        
    # Return the list of booleans.
    return result

def forward_dend(n_queried = n_queried):
    reps, cs, ts = dendrite.text( endpoints=graph.endpoints[ torch.tensor( list(random.sample( range(4096), n_queried )))], inputs = inputs, synapses=synapses, timeout = timeout)
    return [ c.item() == 1 for c in cs ]
##########################
##### Run experiment #####
##########################
def run(forward, n_queried = n_queried, n_tasks = n_tasks):
    # Measure state before.
    start_time = time.time()
    io_1 = psutil.net_io_counters()
    start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv


    # Run each query concurrently then get results as completed.
    exp_results = []
    for i in range(n_trials):
        exp_futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in tqdm(range(n_tasks), desc='Submitting', leave=True):
                exp_futures.append(executor.submit(forward, n_queried))
            for future in exp_futures:
                result = future.result()
                exp_results.append( result )     
            
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
    total_sent = n_queried * n_tasks * n_trials
    total_failed = total_sent - total_success

    total_seconds =  end_time - start_time
    print ('\nElapsed:', total_seconds, 's') 
    print ('\nSteps:', n_tasks ) 
    print ('Step speed:', n_tasks / (total_seconds), "/s" ) 

    print ('\nQueried:', n_queried )
    print ('\nNum Trials:', n_trials )
    print ('\nNum Tasks:', n_tasks )
    print ('Query speed:', total_sent / (total_seconds), "/s" ) 

    print ('\nSuccess', total_success) 
    print ('Failed', total_failed ) 
    print ('Rate', total_success / (total_success + total_failed))

    print("\nTotal Upload:", get_size( total_bytes_sent ))
    print("Total Download:", get_size( total_bytes_recved ))
    print("\nUpload Speed:", get_size( total_bytes_sent/ total_seconds), "/s")
    print("Download Speed:", get_size( total_bytes_recved/ total_seconds), "/s")

    result = {
        'success_rate': total_success / (total_success + total_failed),
        'n_queried': n_queried,
        'avg_seconds': total_seconds / (n_trials), # dont have to be divided by n_task, cuase it was ran in parallel
        'avd_upload': total_bytes_sent / (n_trials * n_tasks),
        'avg_download': total_bytes_recved / (n_trials * n_tasks),
        'n_tasks': n_tasks
    }
    print(result)
    return result

# n_queried_stats = {'dend': [], 'receptor': []}
# for n_queried in range(100, 1100, 100):
#     n_queried_stats['receptor'].append(run(forward, n_queried=n_queried))
#     n_queried_stats['dend'].append(run(forward_dend, n_queried = n_queried))
#     torch.save(n_queried_stats, 'speed_test_n_queried.pt')

n_tasks_stats = {'dend': [], 'receptor': []}
for n_tasks in range(1, 10, 1):
    n_tasks_stats['receptor'].append(run(forward, n_tasks=n_tasks))
    n_tasks_stats['dend'].append(run(forward_dend, n_tasks = n_tasks))
    torch.save(n_tasks_stats, 'speed_test_n_tasks.pt')
