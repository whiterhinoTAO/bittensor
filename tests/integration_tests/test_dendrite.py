# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import torch
import pytest
import bittensor
from multiprocessing import Pool

wallet = bittensor.wallet.mock()
dendrite = bittensor.dendrite( wallet = wallet )
neuron_obj = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    ip = '0.0.0.0',
    ip_type = 4,
    port = 12345,
    hotkey = dendrite.wallet.hotkey.ss58_address,
    coldkey = dendrite.wallet.coldkey.ss58_address,
    modality = 0
)

synapses = [bittensor.synapse.TextLastHiddenState(),bittensor.synapse.TextCausalLM(), bittensor.synapse.TextSeq2Seq(num_to_generate=70)]

def check_resp_shape(resp, num_resp, block_size, seq_len):
    assert len(resp) == num_resp
    assert list(resp[0][0].shape) == [block_size, seq_len, bittensor.__network_dim__]
    assert list(resp[0][1].shape) == [block_size, seq_len, bittensor.__vocab_size__]
    assert list(resp[0][2].shape) == [block_size, 70]
    
def test_dendrite_forward_text_endpoints_tensor():
    endpoints = neuron_obj.to_tensor()
    x = torch.tensor( [[ 1,2,3 ], [ 1,2,3 ]] )
    resp,  _, _ = dendrite.text( endpoints = endpoints, inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 2, seq_len = 3 )
    assert dendrite.stats.total_requests == 1
    dendrite.to_wandb()

def test_dendrite_forward_text_multiple_endpoints_tensor():
    endpoints_1 = neuron_obj.to_tensor()
    endpoints_2 = neuron_obj.to_tensor()
    endpoints = torch.stack( [endpoints_1, endpoints_2], dim=0)
    x = torch.tensor( [[ 1,2,3 ], [ 1,2,3 ]] )
    resp,  _, _ = dendrite.text( endpoints = endpoints, inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 2, seq_len = 3 )

def test_dendrite_forward_text_multiple_endpoints_tensor_list():
    endpoints_1 = neuron_obj.to_tensor()
    endpoints_2 = neuron_obj.to_tensor()
    endpoints_3 = neuron_obj.to_tensor()
    endpoints = [torch.stack( [endpoints_1, endpoints_2], dim=0), endpoints_3]
    x = torch.tensor( [[ 1,2,3 ], [ 1,2,3 ]] )
    resp,  _, _ = dendrite.text( endpoints = endpoints, inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 3, block_size = 2, seq_len = 3 )    

def test_dendrite_forward_text_singular():
    x = torch.tensor( [[ 1,2,3 ], [ 1,2,3 ]] )
    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 2, seq_len = 3 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = [x], synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 2, seq_len = 3 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 2, seq_len = 3 )

    with pytest.raises(ValueError):
        dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = [x], synapses = synapses )

def test_dendrite_forward_text_singular_no_batch_size():
    x = torch.tensor( [ 1,2,3 ] )
    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 1, seq_len = 3 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = [x], synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 1, seq_len = 3 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 1, seq_len = 3 )

    with pytest.raises(ValueError):
        dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = [x], synapses = synapses )

def test_dendrite_forward_text_tensor_list_singular():
    x = [ torch.tensor( [ 1,2,3 ] ) for _ in range(2) ]
    with pytest.raises(ValueError):
        resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 1, seq_len = 3 )

def test_dendrite_forward_text_tensor_list():
    x = [ torch.tensor( [[ 1,2,3 ], [ 1,2,3 ]] ) for _ in range(2) ]
    with pytest.raises(ValueError):
        resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 2, seq_len = 3 )

def test_dendrite_forward_text_singular_string():
    x = "the cat"
    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 1, seq_len = 2 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = [x], synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 1, seq_len = 2 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 1, seq_len = 2 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = [x], synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 1, seq_len = 2 )

def test_dendrite_forward_text_list_string():
    x = ["the cat", 'the dog', 'the very long sentence that needs to be padded']
    resp, _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 3, seq_len = 9 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 3, seq_len = 9 )

def test_dendrite_forward_tensor_shape_error():
    x = torch.rand(3, 3, 3, dtype=torch.float32)
    with pytest.raises(ValueError):
        dendrite.forward_tensor( [neuron_obj], [x])

def test_dendrite_forward_tensor_type_error():
    x = torch.zeros(3, 3, bittensor.__network_dim__, dtype=torch.int32)
    with pytest.raises(ValueError):
        dendrite.forward_tensor( [neuron_obj], x)

def test_dendrite_forward_tensor_endpoint_type_error():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    with pytest.raises(ValueError):
        dendrite.forward_tensor( [dict()], [x])

def test_dendrite_forward_tensor_endpoint_len_error():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    with pytest.raises(ValueError):
        dendrite.forward_tensor( [], [x])

def test_dendrite_forward_tensor_input_len_error():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    with pytest.raises(ValueError):
        dendrite.forward_tensor( [neuron_obj], [])

def test_dendrite_forward_tensor_mismatch_len_error():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    with pytest.raises(ValueError):
        dendrite.forward_tensor( [neuron_obj], [x,x])

def test_dendrite_forward_text_non_list():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops, times = dendrite.text( endpoints = neuron_obj, inputs = x, synapses = synapses )
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [2, 4, bittensor.__network_dim__]

def test_dendrite_forward_text():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops, times = dendrite.text( endpoints = [neuron_obj], inputs = [x], synapses = synapses )
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [2, 4, bittensor.__network_dim__]

def test_dendrite_forward_tensor():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    out, ops, times = dendrite.forward_tensor( [neuron_obj], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]

def test_dendrite_backoff():
    _dendrite = bittensor.dendrite( wallet = wallet )
    _endpoint_obj = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 12345,
        hotkey = _dendrite.wallet.hotkey.ss58_address,
        coldkey = _dendrite.wallet.coldkey.ss58_address,
        modality = 0
    )
    print (_endpoint_obj)
    
    # Normal call.
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    out, ops, times = _dendrite.forward_tensor( [_endpoint_obj], [x])
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable
    assert list(out[0].shape) == [3, 3, bittensor.__network_dim__]
    del _dendrite

def test_dendrite_multiple():
    endpoint_obj = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 12345,
        hotkey = wallet.hotkey.ss58_address,
        coldkey = wallet.coldkey.ss58_address,
        modality = 0
    )
    x = torch.tensor( [ 1,2,3 ] )

    config = bittensor.dendrite.config()
    receptor_pool = bittensor.receptor_pool( 
        wallet = wallet,
        max_worker_threads = config.dendrite.max_worker_threads,
        max_active_receptors = config.dendrite.max_active_receptors,
        compression = config.dendrite.compression,
    )

    authkey = wallet.hotkey.ss58_address.encode('UTF-8')
    manager_server = bittensor.dendrite.manager_serve(config, wallet, receptor_pool, authkey = authkey)

    dend1 = bittensor.dendrite( wallet = wallet, multiprocess=True)
    dend2 = bittensor.dendrite( wallet = wallet, multiprocess=True)
    dend3 = bittensor.dendrite( wallet = wallet, multiprocess=True)
    dend4 = bittensor.dendrite( wallet = wallet, multiprocess=True)
    
    out, ops, times = dend1.forward_text( endpoint_obj, x )
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable

    out, ops, times = dend2.forward_text( endpoint_obj, x )
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable

    out, ops, times = dend3.forward_text( endpoint_obj, x )
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable

    out, ops, times = dend4.forward_text( endpoint_obj, x )
    assert ops[0].item() == bittensor.proto.ReturnCode.Unavailable

    assert len(receptor_pool.receptors) == 1 

    assert manager_server.connected_count == 4

    dend4.__del__()

    assert manager_server.connected_count == 3

    dend3.__del__()

    assert manager_server.connected_count == 2

    dend2.__del__()

    assert manager_server.connected_count == 1

    dend1.__del__()


def test_dendrite_to_df():
    dendrite.to_dataframe(bittensor.metagraph(_mock=True).sync())

def test_dend_del():
    dendrite.__del__()
    
if __name__ == "__main__":
    bittensor.logging(debug = True)
    # test_dendrite_multiple()
    test_dendrite_forward_text_endpoints_tensor()