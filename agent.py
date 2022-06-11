import numpy as np
import torch


class Agent:

    def __init__(self, network_file):
        self.network = torch.load(network_file)
        self.network.eval()
    

    # Send the state through the network and get a distribution over actions
    # Multiply this elementwise with the list of valid actions to remove invalid actions from being selected
    # Valid actions are 1 if valid and 0 if not valid at the index of the action
    # Return the argmax of this distribution as the chosen action
    def get_action(self, state, actions, stochastic):
        with torch.no_grad(): # Disable learning
            x = torch.as_tensor(state, dtype=torch.float32)
            y = self.network(x)
            distribution = np.exp(y.numpy()) # Reverse the logarithm from the network outputs

        distribution *= actions # Elementwise multiplication
        distribution /= np.sum(distribution)

        if stochastic:
            return np.random.choice(len(actions), p=distribution)
        else:
            return np.argmax(distribution)
