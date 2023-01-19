import bittensor 
import concurrent
sub = bittensor.subtensor(network = 'finney')
count = 0
dev_id = 0
n_workers = 20
n_tasks = 100

def reg(wallet_id):
    wallet = bittensor.wallet( name = 'finney_isa', hotkey= f'default_{dev_id}_{wallet_id}')
    wallet.create()
    sub.register(
        wallet = wallet,
        netuid = 0,
    )
    print('device {dev_id}: ', wallet)

while True:
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        for idx, call_arg in enumerate(list(range(count + n_tasks))):
            future = executor.submit(reg, call_arg)
    count += n_tasks