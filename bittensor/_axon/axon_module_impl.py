""" Implementation of Axon, services Forward and Backward requests from other neurons.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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

import sys
import time as clock
from types import SimpleNamespace
from typing import List, Tuple, Callable

import torch
import grpc
import wandb
import pandas
from loguru import logger
import torch.nn.functional as F
import concurrent

from prometheus_client import Counter, Histogram, Enum, CollectorRegistry

import bittensor
import bittensor.utils.stats as stat_utils
from datetime import datetime

logger = logger.opt(colors=True)

class AxonModule(object):
    def __init__( self):
        pass
    def __call__(self ,**kwargs):
        return kwargs


    def default_forward_callback(self, inputs_x:torch.FloatTensor, synapses=[], hotkey = None):
        """
            The default forward callback when no callback is attached: Is used to call specific synapse functions

            Args:
                inputs_x (:obj:`torch.FloatTensor`, `required`): 
                    The inputs that will be passed to the synapse functions
                
                synapses (:obj: list of bittensor.proto.SynapseArgs, 'Optional')
                    The proto message that contains additional args for individual synapse functions

                hotkey (:obj: str of the hotkey, 'Optional')
                    The hotkey of the validator who sent the request

            Returns:
                response_tensors: (:obj: list of bittensor.proto.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                response_codes: (:obj: list of bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
                response_messages: (:obj: list of strings, `required`)
                    return message associated with synapse call
        """
        # --- initialize response variables --- 
        response_tensors = []
        response_codes = []
        response_messages = []
        model_output = None
        
        # --- calling attached synapses ---
        for index, synapse in enumerate(synapses):
            try:
                synapse_check =  self.synapse_checks(synapse, hotkey)

                if synapse.synapse_type in self.synapse_callbacks and self.synapse_callbacks[synapse.synapse_type] != None and synapse_check:
                    message, model_output, response_tensor = self.synapse_callbacks[synapse.synapse_type](inputs_x[index], synapse, model_output)
                    response_tensors.append(response_tensor)
                    response_codes.append(bittensor.proto.ReturnCode.Success)
                    response_messages.append('Success' if message is None else message)
                
                elif not synapse_check:
                    response_tensors.append(None)
                    response_codes.append(bittensor.proto.ReturnCode.UnknownException)
                    response_messages.append('Synapse Check Failed')

                else:
                    response_tensors.append(None)
                    response_codes.append(bittensor.proto.ReturnCode.NotImplemented)
                    response_messages.append('Not Implemented')

            except Exception as e: 
                # --- Exception Hit in Synapse ---
                response_tensors.append(None)
                response_codes.append(bittensor.proto.ReturnCode.UnknownException)
                response_messages.append(str(e))
        
        return response_tensors, response_codes, response_messages

    def default_backward_callback(self, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses=[] ):
        """
            The default backward callback when no callback is attached: Is used to call specific synapse functions

            Args:
                inputs_x (:obj:`torch.FloatTensor`, `required`): 
                    The inputs that will be passed to the synapse functions
                grads_dy (:obj:`torch.FloatTensor`, `required`):
                    The gradients that will be passed to the synapse functions
                synapses (:obj: list of bittensor.proto.SynapseArgs, 'Optional')
                    The proto message that contains additional args for individual synapse functions

            Returns:
                response_tensors: (:obj: list of bittensor.proto.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                response_codes: (:obj: list of bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
                response_messages: (:obj: list of strings, `required`)
                    return message associated with synapse call
        """
        # --- initialize response variables --- 
        response_tensors = []
        response_codes = []
        response_messages = []
        
        # --- calling attached synapses ---
        with torch.enable_grad() and torch.autograd.set_detect_anomaly(True):
            for index, synapse in enumerate(synapses):
                try:
                    if synapse.synapse_type in self.synapse_callbacks and self.synapse_callbacks[synapse.synapse_type] != None:
                        message, model_output, response_tensor = self.synapse_callbacks[synapse.synapse_type](inputs_x[index], synapse)
                        torch.autograd.backward (
                            tensors = [ response_tensor ],
                            grad_tensors = [ grads_dy[index] ],
                            retain_graph=True
                        )                        


                        response_tensors.append(None)
                        response_codes.append(bittensor.proto.ReturnCode.NotImplemented)
                        response_messages.append('Not Implemented')
                except Exception as e:
                    # --- Exception Hit in Synapse ---
                    response_tensors.append(None)
                    response_codes.append(bittensor.proto.ReturnCode.UnknownException)
                    response_messages.append(str(e))

        if self.optimizer_step != None:
            self.optimizer_step()
        
        return response_tensors, response_codes, response_messages
