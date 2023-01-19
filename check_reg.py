import bittensor 
dev_id = 0
reg_count = 0
faulty_reg = 0
meta = bittensor.metagraph(network = 'finney').sync(netuid = 1)
for i in range(3000):
    wallet = bittensor.wallet( name = 'finney_isa', hotkey= f'default_{dev_id}_{i}')
    if wallet.hotkey.ss58_address in meta.hotkeys:
        print(wallet, meta.hotkey_to_uid(wallet.hotkey.ss58_address))
        # reg_count += 1
    # else:
        # faulty_reg += 1

# print(reg_count, faulty_reg)

# print([ obj.hotkey for obj in meta.endpoint_objs if  obj.coldkey == '5CwSXCY5PnuruEme2iwdx5c9zKFtsb9MaQDwHVFU8fdnCiWq'])