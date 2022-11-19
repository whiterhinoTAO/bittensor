import bittensor
import torch.multiprocessing as mp
import torch

import transformers
from transformers import AutoModel,AutoTokenizer,AutoConfig, AutoModelForCausalLM

def load_pretrained_model(config):
    pretrained_model = AutoModelForCausalLM.from_pretrained(config.neuron.model_name)
    if config.neuron.autocast and config.neuron.device[:4] == 'cuda':
        pretrained_model.half()

    return pretrained_model.to(config.neuron.device)


def processfn(queue, config):
    bittensor.neurons.core_server.neuron(queue=queue, config=config).run()

if __name__ == "__main__":
    mp.set_start_method("spawn")

    bittensor.utils.version_checking()
    
    config = bittensor.neurons.core_server.server.config()
    wallet_hotkey = config.wallet.hotkey
    axon_port = config.axon.port
    pretrained_model = load_pretrained_model(config=config)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()

    instances_count = config.create_instances
    instances = []
    for i in range(instances_count):
        queue.put(pretrained_model)

        config.wallet.hotkey = wallet_hotkey + str(i + 1 + config.wallet.hotkey_id)
        config.axon.port = axon_port + i

        instance = mp.Process(target=processfn, args=(queue,config))
        instances.append(instance)
        instance.start()

    for instance in instances:
        instance.join()
