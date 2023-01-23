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

""" Template server.

Example:
    $ import neurons
    $ neurons.text.core_server.neuron().run()
"""

import bittensor
import os

import torch

import argparse 
import json

import sys
from loguru import logger; logger = logger.opt(colors=True)

import random
import numpy as np

import deepspeed
from deepspeed.pipe import PipelineModule


from .nucleus_impl import server
from .run import serve

class neuron:
    r"""
    Creates a bittensor neuron that specializes in the serving. The template server miner
    serves a NLP model from huggingface on the bittensor network. By default, the model does 
    not train itself and thus requires less memory to run. 

    Args: 
            config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
            subtensor (:obj:bittensor.subtensor , `optional`):
                bittensor subtensor connection
            wallet (:obj:bittensor.wallet, `optional`):
                bittensor wallet object
            axon (:obj:bittensor.axon, `optional`):
                bittensor axon object
            metagraph (:obj:bittensor.metagraph, `optional`):
                bittensor metagraph object
            lasthidden (:obj:bool, `optional`):
                lasthidden synapse control
            causallm (:obj:bool, `optional`):
                causallm synapse control
            causallmnext (:obj:bool, `optional`):
                causallmnext synapse control
            seq2seq (:obj:bittensor.metagraph, `optional`):
                seq2seq synapse control
            synapse_list (:obj:list of int, `optional`):
                

    Examples:: 
            >>> subtensor = bittensor.subtensor(network='nakamoto')
            >>> server = bittensor.neuron.text.core_server.neuron(subtensor=subtensor)
            >>> server.run()
    """
    def __init__(
        self, 
        config: 'bittensor.config' = None,
        subtensor: 'bittensor.subtensor' = None,
        wallet: 'bittensor.wallet' = None,
        axon: 'bittensor.axon' = None,
        metagraph: 'bittensor.metagraph' = None,
        lasthidden = None,
        causallm = None,
        causallmnext = None,
        seq2seq = None,
        synapse_list = None,
    ):
        if config == None: config = server.config()
        config = config; 

        if synapse_list != None:
            config.neuron.lasthidden = False
            config.neuron.causallm = False
            config.neuron.causallmnext = False
            config.neuron.seq2seq = False

            if bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE in synapse_list:
                config.neuron.lasthidden = True
            
            if bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM in synapse_list:
                config.neuron.causallm = True

            if bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT in synapse_list:
                config.neuron.causallmnext = True

            if bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ in synapse_list:
                config.neuron.seq2seq = True

        config.neuron.lasthidden = lasthidden if lasthidden != None else config.neuron.lasthidden
        config.neuron.causallm = causallm if causallm != None else config.neuron.causallm
        config.neuron.causallmnext = causallmnext if causallmnext is not None else config.neuron.causallmnext
        config.neuron.seq2seq = seq2seq if seq2seq != None else config.neuron.seq2seq

        self.check_config( config )
        bittensor.logging (
            config = config,
            logging_dir = config.neuron.full_path,
        )
        # Init prometheus.
        # By default we pick the prometheus port to be axon.port - 1000 so that we can match port to server.
        bittensor.prometheus ( 
            config = config,
            port = config.prometheus.port if config.axon.port == bittensor.defaults.axon.port else config.axon.port - 1000
        )

        self.model = server(config = config)
        self.config = config
        self.config.to_prometheus()

        self.subtensor = subtensor
        self.wallet = wallet
        self.axon = axon
        self.metagraph = metagraph
        # world_size = int(os.getenv('WORLD_SIZE', '1'))
        # local_rank = int(os.getenv('LOCAL_RANK', '0'))
        # self.device = torch.device("cuda", local_rank)

        # ds_engine = deepspeed.init_inference(self.model,
        #                          mp_size=world_size,
        #                          dtype=torch.half,
        #                         #  replace_method='auto',
        #                         #  replace_with_kernel_inject=True
        #                          )

        # self.model_engine = ds_engine.module

        # ds_args = config.deepspeed
        # deepspeed.init_distributed()

        # self.net = PipelineModule(layers=[self.model], num_stages=1)
        # self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
        #     args = ds_args,
        #     model = self.model,
        #     # model = self.net,
        #     model_parameters = self.model.parameters(),
        #     # training_data = self.dataset
        # )
        # self.device = torch.device('cuda', ds_args.local_rank)


    def run(self):
        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        serve(
            self.config,
            self.model,
            subtensor = self.subtensor,
            wallet = self.wallet,
            axon = self.axon,
            metagraph = self.metagraph,
        )


    def simple_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file', type=str, help='DS config file.', default='~/.bittensor/bittensor/_neuron/text/core_server/ds_config.json')
        parser.add_argument('--local_rank', type=int, help='local rank', default=0)
        args = parser.parse_args()
        args.deepspeed_config = args.config_file
        args.config = json.load(open(args.config_file, 'r', encoding='utf-8'))
        return args


    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file', type=str, help='DS config file.')
        # Include DeepSpeed configuration arguments
        parser = deepspeed.add_config_arguments(parser)

        args = parser.parse_args()

        # no cuda mode is not supported
        args.no_cuda = False
        args.job_name = 'ds_test'
        args.output_dir = os.path.expanduser('~/.bittensor/ds_run/')
        args.seed = 0
        args.validation_data_path_prefix = None
        args.max_steps = 10
        args.max_steps_per_epoch = 5
        args.deepspeed_config = args.config_file

        return args

    def construct_args(self):
        args = self.get_args()

        # Prepare Logger
        config = json.load(open(args.config_file, 'r', encoding='utf-8'))

        # # choose dataset and training config based on the given sequence length
        # seq_len = str(args.max_seq_length)

        # datasets = config["data"]["mixed_seq_datasets"][seq_len]
        # del config["data"]["mixed_seq_datasets"]
        # training = config["mixed_seq_training"][seq_len]
        # del config["mixed_seq_training"]
        # config["data"]["datasets"] = datasets
        # config["training"] = training
        args.config = config

        args.job_name = config['name'] if args.job_name is None else args.job_name
        print("Running Config File: ", args.job_name)
        # Setting the distributed variables
        print("Args = {}".format(args))

        # Setting all the seeds so that the task is random but same accross processes
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        os.makedirs(args.output_dir, exist_ok=True)
        args.saved_model_path = os.path.join(args.output_dir, "saved_models/",
                                            args.job_name)

        args.n_gpu = 1

        # Loading Tokenizer
        tokenizer = bittensor.tokenizer()
        args.tokenizer = tokenizer

        # Set validation dataset path
        if args.validation_data_path_prefix is None:
            logger.warning(
                'Skipping validation because validation_data_path_prefix is unspecified'
            )

        # Issue warning if early exit from epoch is configured
        if args.max_steps < sys.maxsize:
            logger.warning(
                'Early training exit is set after {} global steps'.format(
                    args.max_steps))

        if args.max_steps_per_epoch < sys.maxsize:
            logger.warning('Early epoch exit is set after {} global steps'.format(
                args.max_steps_per_epoch))

        return 

    @classmethod
    def config(cls):
        return server.config()

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.wandb.check_config( config )
        bittensor.prometheus.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.get('name', bittensor.defaults.wallet.name), config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)
