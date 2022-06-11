import numpy as np
import torch


class ActorNet():

    def __init__(self, layer_nodes, activation_functions, learning_rate, optimizer, network_file=None):
        if network_file is not None:
            self.model = torch.load(network_file) # Load a previous model to continue training with that
        else:
            modules = []
            for i in range(len(layer_nodes) - 2):
                modules.append(torch.nn.Linear(layer_nodes[i], layer_nodes[i + 1]))
                if activation_functions[i] == "linear":
                    pass
                elif activation_functions[i] == "sigmoid":
                    modules.append(torch.nn.Sigmoid())
                elif activation_functions[i] == "tanh":
                    modules.append(torch.nn.Tanh())
                elif activation_functions[i] == "relu":
                    modules.append(torch.nn.ReLU())
            modules.append(torch.nn.Linear(layer_nodes[-2], layer_nodes[-1]))
            modules.append(torch.nn.LogSoftmax(dim=0)) # Need a logarithm to work correctly with KLDivergence loss in pytorch
            self.model = torch.nn.Sequential(*modules)

        self.criterion = torch.nn.KLDivLoss(reduction="batchmean") # Useful for comparing probability distributions

        if optimizer == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    

    # Do a forward pass through the network
    def predict(self, state):
        with torch.no_grad(): # Disable learning for the predictions
            x = torch.as_tensor(state, dtype=torch.float32) # Convert tuple to torch tensor
            y = self.model(x)
            distribution = np.exp(y.numpy()) # Because the network outputs logarithms, reverse this here
            return distribution # Should still sum to 1, as the logarithm is applied after a softmax


    # Update the parameters in the network by training on cases from the replay buffer
    # Take the whole replay buffer as a parameter and a set of indexes for the current randomly sampled batch
    # First create torch tensor with the relevant data for training
    # Then do the actual backwards pass
    # Send the data through the model
    # Compute the loss and call backward to compute gradients
    # The optimizer will use the gradients to update the parameters
    def train(self, replay_buffer, batch_indexes):
        x = torch.zeros((len(batch_indexes), len(replay_buffer[0][0])), dtype=torch.float32)
        y_true = torch.zeros((len(batch_indexes), len(replay_buffer[0][1])), dtype=torch.float32)
        for i in range(len(batch_indexes)):
            x[i, :] = torch.as_tensor(replay_buffer[batch_indexes[i]][0])
            y_true[i, :] = torch.as_tensor(replay_buffer[batch_indexes[i]][1])
        
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        loss.backward()
        self.optimizer.step()
    

    # Save the model to a file
    def save(self, episode):
        torch.save(self.model, "models/model" + str(episode))
