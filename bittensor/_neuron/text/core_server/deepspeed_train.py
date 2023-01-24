from bittensor._neuron.text.core_server.nucleus_impl import server
import bittensor 
import torch
import deepspeed
import argparse
import json
import random
import os
import sys
from loguru import logger
from transformers import AutoTokenizer
import numpy as np
from deepspeed.pipe import PipelineModule
from transformers import AutoModel,AutoTokenizer,AutoConfig, AutoModelForCausalLM
import time 

logger = logger.opt(colors=True)

class DeepSpeedTrain:
    def __init__(self):
        self.dataset = bittensor.dataset(max_directories = 10)
        next(self.dataset)
        self.use_net = False
        ds_args = self.simple_args()
        deepspeed.init_distributed()
        print(ds_args)
        if self.use_net:
            model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
            net = PipelineModule(layers=[model], num_stages=1)
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                args = ds_args,
                model = net,
                model_parameters = [p for p in model.parameters() if p.requires_grad],
                training_data = self.dataset
            )
        else:
            model = server(
                model_name = 'EleutherAI/gpt-neo-2.7B'
            )
            # model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
            # model.pre_model.train()
            # model.set_fine_tuning_params
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                args = ds_args,
                model = model,
                model_parameters = model.parameters(),
                training_data = self.dataset
            )
        
        # =================
        self.device = torch.device('cuda', ds_args.local_rank)
        self.ema_decay = 0.97
        self.path = os.path.expanduser('~/.bittensor/bittensor')

    def run(self):
        print('\n\nstart run\n\n')
        stats = {
            'losses': [],
            'times': [],
            'steps': 0,
            'ema_loss': None,
            'best_loss': float('inf')
        }
        while True:
            start_time = time.time()
            if self.use_net:
                loss = self.model_engine.train_batch()
            else:
                self.model_engine.train()
                loss, decoded_targe = self.model_engine(next(self.dataset).to(self.device))
                self.model_engine.backward(loss)
                self.model_engine.step()

            if stats['ema_loss'] == None:
                stats['ema_loss'] = loss.detach().item()
            stats['ema_loss'] = self.ema_decay * stats['ema_loss'] + (1- self.ema_decay) * loss.detach().item() 
            stats['losses'].append(loss.detach().item())
            stats['times'].append (time.time() - start_time)
            stats['steps'] += 1

            if stats['steps'] % 100 == 0:
                print( f"{self.device}, step: {stats['steps']}, ema_loss: {stats['ema_loss']}, best_loss: {stats['best_loss']}")
                print(stats['ema_loss'], stats['best_loss'], (stats['ema_loss'] < stats['best_loss']))
                torch.save(stats, 'deepspeed_train.pt')
                if self.device == 'cuda:0' and (stats['ema_loss'] < stats['best_loss']):
                    self.model_engine.save_checkpoint(self.path, stats['ema_loss'], client_sd = stats['steps'])
                    print(f"Saved mode: loss {stats['best_loss']} -> {stats['ema_loss']}")
                    stats['best_loss'] = stats['ema_loss']

    def simple_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file', type=str, help='DS config file.')
        parser.add_argument('--local_rank', type=int, help='local rank')
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

        return args


if __name__ == "__main__":
    ds = DeepSpeedTrain()
    ds.run()
    ds.dataset.close()

