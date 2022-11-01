"""
Implementation of the config class, which manages the config of different bittensor modules.
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


import os
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Optional
import yaml
import json
from munch import Munch
from prometheus_client import Info
from pandas.io.json import json_normalize
import bittensor
from copy import deepcopy
from .utils import dict_put, dict_get, dict_fn, dict_fn_local_copy, dict_fn_get_config 


class Config ( Munch ):
    """
    Implementation of the config class, which manages the config of different bittensor modules.
    """
    def __init__(self, loaded_config = None ):
        super().__init__()
        if loaded_config:
            raise NotImplementedError('Function load_from_relative_path is not fully implemented.')



    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return "\n" + yaml.dump(self.toDict())

    def to_string(self, items) -> str:
        """ Get string from items
        """
        return "\n" + yaml.dump(items.toDict())

    def update_with_kwargs( self, kwargs ):
        """ Add config to self
        """
        for key,val in kwargs.items():
            self[key] = val

    def to_prometheus(self):
        """
            Sends the config to the inprocess prometheus server if it exists.
        """
        try:
            prometheus_info = Info('config', 'Config Values')
            config_info = json_normalize(json.loads(json.dumps(self)), sep='.').to_dict(orient='records')[0]
            formatted_info = {}
            for key in config_info:
                config_info[key] = str(config_info[key])
                formatted_info[key.replace('.', '_')] = str(config_info[key])
            prometheus_info.info(formatted_info)
        except ValueError:
            # The user called this function twice in the same session.
            # TODO(const): need a way of distinguishing the various config items.
            bittensor.__console__.print("The config has already been added to prometheus.", highlight=True)

    def to_defaults(self):
        try: 
            if 'axon' in self.keys():
                bittensor.defaults.axon.port = self.axon.port
                bittensor.defaults.axon.ip = self.axon.ip
                bittensor.defaults.axon.external_port = self.axon.external_port
                bittensor.defaults.axon.external_ip = self.axon.external_ip
                bittensor.defaults.axon.max_workers = self.axon.max_workers
                bittensor.defaults.axon.maximum_concurrent_rpcs = self.axon.maximum_concurrent_rpcs
            
            if 'dataset' in self.keys():
                bittensor.defaults.dataset.batch_size = self.dataset.batch_size
                bittensor.defaults.dataset.block_size = self.dataset.block_size
                bittensor.defaults.dataset.num_batches = self.dataset.num_batches
                bittensor.defaults.dataset.num_workers = self.dataset.num_workers
                bittensor.defaults.dataset.dataset_name = self.dataset.dataset_name
                bittensor.defaults.dataset.data_dir = self.dataset.data_dir
                bittensor.defaults.dataset.save_dataset = self.dataset.save_dataset
                bittensor.defaults.dataset.max_datasets = self.dataset.max_datasets

            if 'dendrite' in self.keys():
                bittensor.defaults.dendrite.timeout = self.dendrite.timeout
                bittensor.defaults.dendrite.max_worker_threads = self.dendrite.max_worker_threads
                bittensor.defaults.dendrite.max_active_receptors = self.dendrite.max_active_receptors
                bittensor.defaults.dendrite.requires_grad = self.dendrite.requires_grad

            if  'logging' in self.keys():
                bittensor.defaults.logging.debug = self.logging.debug
                bittensor.defaults.logging.trace = self.logging.trace
                bittensor.defaults.logging.record_log = self.logging.record_log
                bittensor.defaults.logging.logging_dir = self.logging.logging_dir
            
            if 'subtensor' in self.keys():
                bittensor.defaults.subtensor.network = self.subtensor.network
                bittensor.defaults.subtensor.chain_endpoint = self.subtensor.chain_endpoint
            
            if 'threadpool' in self.keys():
                bittensor.defaults.threadpool.max_workers = self.threadpool.max_workers
                bittensor.defaults.threadpool.maxsize = self.threadpool.maxsize 

            if 'wallet' in self.keys():
                bittensor.defaults.wallet.name = self.wallet.name
                bittensor.defaults.wallet.hotkey = self.wallet.hotkey
                bittensor.defaults.wallet.path = self.wallet.path
            
            if 'wandb' in self.keys():
                bittensor.defaults.wandb.name = self.wandb.name
                bittensor.defaults.wandb.project = self.wandb.project
                bittensor.defaults.wandb.tags = self.wandb.tags
                bittensor.defaults.wandb.run_group = self.wandb.run_group
                bittensor.defaults.wandb.directory = self.wandb.directory
                bittensor.defaults.wandb.offline = self.wandb.offline

        except Exception as e:
            print('Error when loading config into defaults {}'.format(e))


    default_dict_fns = {
        'local_copy': dict_fn_local_copy,
        'get_config': dict_fn_get_config
    }

    @staticmethod
    def dict_fn(fn, input, context=None, seperator='::', default_dict_fns={}):
        if len(default_dict_fns) == 0:
            default_dict_fns = Config.default_dict_fns()
        if context == None:
            context = deepcopy(context)
        
        if type(input) in [dict]:
            keys = list(input.keys())
        elif type(input) in [set, list, tuple]:
            input = list(input)
            keys = list(range(len(input)))
        
        for key in keys:
            if isinstance(input[key], str):
                if len(input[key].split(seperator)) == 2: 
                    function_key, input_arg =  input[key].split(seperator)
                    input[key] = default_dict_fns[function_key](input=input, context=context)
            
            input[key] = dict_fn(fn=fn, 
                                    input=input, 
                                    context=context,
                                    seperator=seperator,
                                    default_dict_fns=default_dict_fns)
    
        return input



    """
    Create and init the config class, which manages the config of different bittensor modules.
    """
    class InvalidConfigFile(Exception):
        """ In place of YAMLError
        """

    def __new__( cls, parser: ArgumentParser = None, strict: bool = False, args: Optional[List[str]] = None ):
        r""" Translates the passed parser into a nested Bittensor config.
        Args:
            parser (argparse.Parser):
                Command line parser object.
            strict (bool):
                If true, the command line arguments are strictly parsed.
            args (list of str):
                Command line arguments.
        Returns:
            config (bittensor.Config):
                Nested config object created from parser arguments.
        """
        if parser == None:
            parser = ArgumentParser()

        # Optionally add config specific arguments
        try:
            parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')
        except:
            # this can fail if the --config has already been added.
            pass
        try:
            parser.add_argument('--strict',  action='store_true', help='''If flagged, config will check that only exact arguemnts have been set.''', default=False )
        except:
            # this can fail if the --config has already been added.
            pass

        # Get args from argv if not passed in.
        if args == None:
            args = sys.argv[1:]

        # 1.1 Optionally load defaults if the --config is set.
        try:
            config_file_path = str(os.getcwd()) + '/' + vars(parser.parse_known_args(args)[0])['config']
        except Exception as e:
            config_file_path = None

        # Parse args not strict
        params = cls.__parse_args__(args=args, parser=parser, strict=False)

        # 2. Optionally check for --strict, if stict we will parse the args strictly.
        strict = params.strict
                        
        if config_file_path != None:
            config_file_path = os.path.expanduser(config_file_path)
            try:
                with open(config_file_path) as f:
                    params_config = yaml.safe_load(f)
                    print('Loading config defaults from: {}'.format(config_file_path))
                    parser.set_defaults(**params_config)
            except Exception as e:
                print('Error in loading: {} using default parser settings'.format(e))

        # 2. Continue with loading in params.
        params = cls.__parse_args__(args=args, parser=parser, strict=strict)

        _config = Config()

        # Splits params on dot syntax i.e neuron.axon_port            
        for arg_key, arg_val in params.__dict__.items():
            split_keys = arg_key.split('.')
            head = _config
            keys = split_keys
            while len(keys) > 1:
                if hasattr(head, keys[0]):
                    head = getattr(head, keys[0])  
                    keys = keys[1:]   
                else:
                    head[keys[0]] = Config()
                    head = head[keys[0]] 
                    keys = keys[1:]
            if len(keys) == 1:
                head[keys[0]] = arg_val

        return _config

    @staticmethod
    def __parse_args__( args: List[str], parser: ArgumentParser = None, strict: bool = False) -> Namespace:
        """Parses the passed args use the passed parser.
        Args:
            args (List[str]):
                List of arguments to parse.
            parser (argparse.ArgumentParser):
                Command line parser object.
            strict (bool):
                If true, the command line arguments are strictly parsed.
        Returns:
            Namespace:
                Namespace object created from parser arguments.
        """
        if not strict:
            params = parser.parse_known_args(args=args)[0]
        else:
            params = parser.parse_args(args=args)

        return params

    @staticmethod
    def full():
        """ From the parser, add arguments to multiple bittensor sub-modules
        """
        parser = ArgumentParser()
        bittensor.wallet.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.prometheus.add_args( parser )
        return bittensor.config( parser )


    