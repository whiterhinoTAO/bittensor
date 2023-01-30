#!/bin/bash
export TOKENIZERS_PARALLELISM=true
export NCCL_P2P_DISABLE=1
python3 ./bittensor/_neuron/text/core_server/local_train.py --config_file ./bittensor/_neuron/text/core_server/ds_config.json
