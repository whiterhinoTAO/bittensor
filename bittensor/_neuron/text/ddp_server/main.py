
import bittensor

if __name__ == "__main__":
    # Setup the subtensor.
    bittensor.neurons.ddp_server.neuron().run()