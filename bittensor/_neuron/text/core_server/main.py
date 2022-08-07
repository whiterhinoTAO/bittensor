import bittensor
if __name__ == "__main__":
    bittensor.utils.version_checking()
    bittensor.neurons.core_server.neuron().run()
