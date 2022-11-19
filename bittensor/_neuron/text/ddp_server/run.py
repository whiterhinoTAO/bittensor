#!/bin/python3
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
""" Advanced server neuron.

Example:
    $ python miners/text/multitron_server/main.py

"""
from re import I
import pdb

import bittensor
import torch
import pandas
import datetime
import traceback
import sys
import os

from loguru import logger; logger = logger.opt(colors=True)
from datetime import datetime,timedelta
from threading import Lock

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_

import time
from multiprocessing import Process, Manager, Event 
import threading 



os.environ['TOKENIZERS_PARALLELISM'] = 'false'

torch.autograd.set_detect_anomaly(True) 

class DDPPipe():
    def __init__( self, config: 'bittensor.config', gp_server, wallet: 'bittensor.wallet', forward_q, events, outputs):
        r""" Initializes the neuron with the passed config.
        """
        torch.autograd.set_detect_anomaly(True) 
        self.config = config
        self.config.to_defaults()
        self.gp_server = gp_server# .to(gp_server.device)
        self.wallet = wallet
        self.world_size = config.neuron.world_size
        self.forward_q = forward_q
        self.events = events
        self.outputs = outputs



    def init_process(self, rank):
        r""" For each process, anchor them to the process group 
        so that they know how to communication with each other.

        Args:
            rank (int):
                rank (id) of the process.
        """
        os.environ['MASTER_ADDR'] = self.config.neuron.address
        os.environ['MASTER_PORT'] = self.config.neuron.port
        if 'cuda' in self.config.neuron.device:
            backend = 'nccl'
        else:
            backend = 'gloo'

        dist.init_process_group(
                backend, 
                rank=rank, 
                world_size=self.world_size, 
        )
    
    def init_bit(self, rank = 0):
        r""" Init bittensor modules after spawning process.
        
        Args:
            rank (int):
                rank (id) of the process.
        """
        self.device = torch.device( device = f'cuda:{rank}' )        
        self.gp_server.device = self.device
        self.gp_server = self.gp_server.to(self.device)
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.metagraph = bittensor.metagraph ( config = self.config, subtensor = self.subtensor )
        self.metagraph.sync()
        self.optimizer = torch.optim.SGD(
            [ {'params': self.gp_server.parameters() } ],
            lr = self.config.neuron.learning_rate,
            momentum = self.config.neuron.momentum,
        )
        
        if rank == 0 :
            logger.success( self.subtensor )
            self.subtensor.register( self.wallet )

        bittensor.tokenizer()
        logger.success( 'Initialized process: {}', rank )

    def cleanup(self):
        r""" Kill the process.
        """
        dist.destroy_process_group()

    def run_parallel( self, ready = True):
        r""" Spawn multiple processes.
        """
        self.process_ctx = mp.spawn(self.run,
            args=(self.world_size, ready),
            nprocs=self.world_size,
            join = True
        )


    def run(self, rank = 0, world_size = 0, ready= None):
        self.init_bit(rank)
        if self.config.neuron.restart == False:
            self.gp_server.load(self.config.neuron.full_path)
        
        self.gp_server = self.gp_server.to(self.device) 

        nn = self.subtensor.neuron_for_pubkey(self.wallet.hotkey.ss58_address)
        uid = nn.uid

        # --- last sync block 
        last_sync_block = self.subtensor.get_current_block()
        last_set_block = last_sync_block
        last_log_block = last_sync_block
        last_log_time = time.time()
        # -- Main Training loop --
        if ready != None and rank == 0 :
            ready.set()

        try:
            torch.cuda.empty_cache()
            while True: 
                try:
                    request_id, inputs_x, synapse = self.forward_q.get(timeout = self.config.neuron.console_log_time)

                    
                    if inputs_x != None:
                        inputs_x = inputs_x.to(self.device)
                        # with self.mutex:
                        message, model_output, topk_token_phrases = self.gp_server.encode_forward_causallmnext(inputs_x,
                                                                                                                    topk=synapse.topk,
                                                                                                                    model_output=None)
                        message_clone = message.detach().clone().cpu()
                        model_output_clone = model_output.detach().clone().cpu()
                        topk_token_phrases_clone = topk_token_phrases.detach().clone().cpu()
                        self.outputs[request_id] = (message_clone, model_output_clone, topk_token_phrases_clone)
                        self.events[request_id].set()
                        
                        # Delete the input tensor to free up memory.
                        del message
                        del model_output
                        del topk_token_phrases
                        del message_clone
                        del model_output_clone
                        del topk_token_phrases_clone

                    del inputs_x
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(e)
                    if 'out of memory' in str(e):                    
                        for p in self.gp_server.pre_model.parameters():
                            if p.grad is not None:
                                del p.grad                    
                        if inputs_x != None:
                            del inputs_x
                        torch.cuda.empty_cache()
                        bittensor.logging.success('cleaned memory', sufix = f'rank: {rank}, {e}')
                
                # log if a certain time period had passed
                # checking with time instead of block here to avoid frequent syncing from subtensor in a while loop
                if time.time() - last_log_time > self.config.neuron.console_log_time:
                    last_log_time = time.time()

                    # ---- syncing metagraph for all rank
                    current_block = self.subtensor.get_current_block()
                    if current_block - last_sync_block > self.config.neuron.metagraph_sync:
                        self.metagraph.sync()
                        last_sync_block = current_block

                    # ---- console logging                    
                    if rank == 0:
                        # ---- data
                        data = {
                            'block': current_block,
                            'stake': nn.stake,
                            'rank': nn.rank,
                            'incentive': nn.incentive,
                            'trust': nn.trust,
                            'consensus': nn.consensus,
                            'incentive': nn.incentive,
                            'dividends': nn.dividends,
                            'emission':  nn.emission,
                        } 
                        
                        # ---- console logging
                        bittensor.__console__.print('[green]Current Status:[/green]', data)

        except Exception as e:
            # --- Unknown error ----
            logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())

class ddp_server:
    def __init__( self, config: 'bittensor.config', gp_server):
        r""" Initializes the neuron with the passed config.
        """
        self.config = config
        self.wallet = bittensor.wallet( config = config ).create().register()
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.metagraph = bittensor.metagraph ( config = self.config, subtensor = self.subtensor )
        
        ctx = mp.get_context('spawn')
        self.forward_q = ctx.Queue()
        logger.info('spawned forward_q')
        
        self.manager = Manager()
        self.events = self.manager.dict()
        self.outputs = self.manager.dict()

        self.gp_server = gp_server
        
        self.axon = bittensor.axon (
            config = self.config,
            wallet = self.wallet,
            synapse_checks=self.synapse_check,
            synapse_causal_lm_next = self.forward_casual_lm_next,
            # blacklist = self.blacklist if not self.gp_server.config.neuron.disable_blacklist else None,
            # priority = self.priority if not self.gp_server.config.neuron.disable_priority else None,
        ) 

        self.optimizer = torch.optim.SGD(
            [ {"params": self.gp_server.parameters()} ],
            lr = config.neuron.learning_rate,
            momentum = config.neuron.momentum,
        )    
        self.axon_pipe = DDPPipe(config, gp_server, self.wallet, self.forward_q, self.events, self.outputs )
        self.timecheck = {}
        self.futures = {}
        self.last_sync_block = None
        self.last_set_weight_block = None

    def forward_casual_lm_next( self, inputs_x: torch.FloatTensor, synapse, model_output=None ):
        r""" Forward function that is called when the axon recieves a forward request from other peers
            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.

                synapse (:obj:`bittensor.synapse`, `required`):
                    The synapse object that is used to forward the request.

                model_output (:obj:`torch.FloatTensor`, `optional`, defaults to None):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]

            Returns:
                message (:obj:`bittensor.proto.ReturnMessage`, `required`):
                    The return message from the nucleus.
                
                model_output (:obj:`torch.FloatTensor`, `required`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
                
                topk_token_phrases (:obj:`torch.FloatTensor`, `required`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        result = None
        request_id = id(inputs_x)


        # logger.info('inputs: ')
        # logger.info(inputs)
    
        self.forward_q.put( (request_id, inputs_x, synapse) )
        self.events[request_id] = self.manager.Event()

        if self.events[request_id].wait(16):
            result = self.outputs[request_id]

        del self.events[request_id]
        del self.outputs[request_id]

        # bittensor.logging.info( 'forward_casual_lm_next: result: {}', result )

        # logger.info('result: ')
        # logger.info(result)
        message = result['message']
        model_output = result['model_output']
        topk_token_phrases = result['topk_token_phrases']

        return message, model_output, topk_token_phrases

    def priority(self, pubkey:str, request_type:bittensor.proto.RequestType, inputs_x) -> float:
        r"""Calculates the priority on requests based on stake and size of input

            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """        
        uid = self.metagraph.hotkeys.index(pubkey)
        priority = self.metagraph.S[uid].item()/ sys.getsizeof(inputs_x)

        return priority

    def blacklist(self, pubkey:str, request_type:bittensor.proto.RequestType) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """

        # Check for stake
        def stake_check() -> bool:
            # If we allow non-registered requests return False = not blacklisted.
            is_registered = pubkey in self.metagraph.hotkeys
            if not is_registered:
                if self.config.neuron.blacklist_allow_non_registered:
                    return False
                else:
                    return True

            # Check stake.
            uid = self.metagraph.hotkeys.index(pubkey)
            if request_type == bittensor.proto.RequestType.FORWARD:
                if self.metagraph.S[uid].item() < self.config.neuron.blacklist.stake.forward:
                    return True
                else:
                    return False

            elif request_type == bittensor.proto.RequestType.BACKWARD:
                if self.metagraph.S[uid].item() < self.config.neuron.blacklist.stake.backward:
                    return True
                else:
                    return False

        # Check for time
        def time_check():
            current_time = datetime.now()
            if pubkey in self.timecheck.keys():
                prev_time = self.timecheck[pubkey]
                if current_time - prev_time >= timedelta(seconds=self.config.neuron.blacklist.time):
                    self.timecheck[pubkey] = current_time
                    return False
                else:
                    self.timecheck[pubkey] = current_time
                    return True
            else:
                self.timecheck[pubkey] = current_time
                return False

        # Black list or not
        if stake_check() or time_check():
            return True
        else: 
            return False

    def synapse_check(self, synapse, hotkey):
        """
            Custom synapse function to protect certain synapse functions depending on the stake and weight.
            Certain synapses require more compute than others. For instance, TEXT_SEQ_2_SEQ requires a significantly
            more commitment by the server than a requeset for TEXT_CAUSAL_LM_NEXT.

            Args:
                synapse (:obj:`bittensor.proto.SynapseArgs`, `required`): 
                    The proto message that contains additional args for individual synapse functions
                hotkey (:obj:`torch.FloatTensor`, `required`):
                    The hotkey that sent the request

        """
        ## Uid that sent the request
        incoming_uid = self.metagraph.hotkeys.index(hotkey)
        if synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE:
            
            if self.metagraph.S[incoming_uid] < self.config.neuron.lasthidden_stake:
                return False
            
        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM:

            if self.metagraph.S[incoming_uid] < self.config.neuron.causallm_stake:
                return False

        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:

            if self.metagraph.S[incoming_uid] < self.config.neuron.causallmnext_stake:
                return False

        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ:

            if (self.metagraph.S[incoming_uid] < self.config.neuron.seq2seq_stake) and (self.metagraph.S[incoming_uid,  self.uid]):
                return False     
        else:
            return False

        return True


    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


    def run(self):
        def serve_when_ready(serve_kwargs, pipe_ready):
            r""" Start to serve Axon when DDP have started
                Args:
                    serve_kwargs(map):
                        Arguments for serving axon.
                    pipe_ready(manager.Event):
                        The Event when the DDP is ready
            """
            if pipe_ready.wait():
                self.axon.start().serve(**serve_kwargs)

                self.axon.optimizer_step = self.optimizer_step
            
            return 
        
        def sync(keyboard_interupt):
            r""" Sync with metagraph and set weight to chain.
                Args:
                    keyboard_interupt(manager.Event):
                        Whether we have tried to stop the program with keyboard_interupt.
            """
            while not keyboard_interupt.is_set():
                current_block = self.subtensor.get_current_block()
                if (self.last_sync_block == None) or (current_block - self.last_sync_block > self.config.neuron.metagraph_sync):
                    self.last_sync_block = current_block
                    self.metagraph.sync()
                    bittensor.logging.success('Metagraph synced', sufix = f'{self.last_sync_block} --> {current_block}')
                    
                if (self.last_set_weight_block == None) or (current_block - self.last_set_weight_block > self.config.neuron.blocks_per_set_weights):
                    self.last_set_weight_block = current_block
                    chain_weights = torch.zeros(self.metagraph.n)
                    chain_weights [ self.uid ] = 1 
                    did_set = self.subtensor.set_weights(
                        uids=self.metagraph.uids,
                        weights = chain_weights,
                        wait_for_inclusion = False,
                        wallet = self.wallet,
                    )
                    
                    if did_set:
                        logger.success('Successfully set weights on the chain')
                    else:
                        logger.error('Failed to set weights on chain. (Timeout)')
                
                time.sleep(self.config.neuron.check_sync_time)
            
        try: 
            neuron = self.subtensor.neuron_for_pubkey(self.wallet.hotkey.ss58_address)
            self.uid = neuron.uid

            pipe_ready = self.manager.Event()
            keyboard_interupt = self.manager.Event()
            axon_start_thread = threading.Thread( target = serve_when_ready, args = ({'subtensor': self.subtensor}, pipe_ready) )
            sync_thread = threading.Thread( target = sync, args = (keyboard_interupt, ))
            axon_start_thread.start()
            sync_thread.start()
            self.axon_pipe.run_parallel(ready = pipe_ready)
            
            # Just to keep this run function alive.
            while True:
                logger.info(pipe_ready)
                time.sleep(20)

        except KeyboardInterrupt:
            keyboard_interupt.set()
            logger.success('Keyboard Interuped')
            self.axon.stop()
            axon_start_thread.join()
            sync_thread.join()
        except Exception as e:
            # --- Unknown error ----
            logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())


