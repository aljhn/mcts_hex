from config_parser import parse_config
from hex import Hex
from actornet import ActorNet
from rl import ReinforcementLearner
from agent import Agent
from tournament import Tournament

# Seeding
import random
import numpy as np
import torch
random.seed(69)
np.random.seed(69)
torch.manual_seed(69)


# Create a game object to handle logic on the game side
# Then create an actor network with the given parameters
# Finally create a reinforcement learner object and use it to train the actor network on the game with the given parameters
def train(config):
    board_size = int(config["board_size"])
    game = Hex(board_size)

    layer_nodes = [(board_size * board_size + 1) * 2] + list(map(int, config["hidden_layer_nodes"])) + [board_size * board_size] # Concatenate the hidden layers with the input and output layers
    activation_functions = config["activation_functions"]
    learning_rate = float(config["learning_rate"])
    optimizer = config["optimizer"]
    actornet = ActorNet(layer_nodes, activation_functions, learning_rate, optimizer)

    reinforcement_learner = ReinforcementLearner(game, actornet)

    actual_games = int(config["actual_games"])
    initial_search_games = int(config["initial_search_games"])
    final_search_games = int(config["final_search_games"])
    save_interval = int(config["save_interval"])
    exploration_constant = float(config["exploration_constant"])
    epsilon = float(config["epsilon"])
    epsilon_decay = float(config["epsilon_decay"])
    visualize = config["mcts_visualize"] == "true"
    visualization_frequency = int(config["visualization_frequency"])
    visualization_frame_delay = float(config["visualization_frame_delay"])
    reinforcement_learner.train(actual_games, initial_search_games, final_search_games, save_interval, visualize, visualization_frequency, visualization_frame_delay, exploration_constant, epsilon, epsilon_decay)


# Create a game object to handle logic on the game side
# Get the amount of agents and load the same amount of models
# Saved model names is based on the save_interval parameter
# Return the results in a 2d-list
# Visualize games if parameter is set
def play(config):
    board_size = int(config["board_size"])
    game = Hex(board_size)

    players = int(config["topp_players"])
    modelnames = config["topp_modelnames"]
    save_interval = int(config["topp_save_interval"])
    agents = []
    for i in range(players):
        agents.append(Agent(modelnames + "/model" + str(save_interval * i)))

    topp = Tournament(agents, game)

    games = int(config["topp_games"])
    results = topp.run(games)
    #results = topp.play({1: 0, 2: 2}, 2)
    print("TOPP results:")
    print(results)

    visualize = config["topp_visualize"] == "true"
    frame_delay = float(config["visualization_frame_delay"])
    if visualize:
        topp.visualize(frame_delay)


if __name__ == "__main__":

    config = parse_config("config.txt")

    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "play":
        play(config)
