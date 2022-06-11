import copy
import numpy as np
from tqdm import tqdm
from mct import MonteCarloTree


class ReinforcementLearner:

    def __init__(self, game, actornet):
        self.game = game
        self.search_game = copy.deepcopy(game) # Create a deepcopy to use inside the MCTS
        self.actornet = actornet


    # Train the actor network on the given game with the pseudocode given in the assignment text
    # Play out multiple games using the distribution found with the MCTS
    # For every move in the every game, do many full search games within the MCTS
    # Save the root state and distribution from the MCTS as a training case for the actor network
    def train(self, actual_games, initial_search_games, final_search_games, save_interval, visualize, visualization_frequency, visualization_frame_delay, exploration_constant, epsilon, epsilon_decay):

        replay_buffer = []

        self.actornet.save(0)

        for g_a in tqdm(range(actual_games)):
            start_player = 1 if g_a % 2 == 0 else 2 # Alternate which player begins
            self.game.initialize(player=start_player)

            tree = MonteCarloTree(self.game.get_state(), self.game.player, self.search_game, self.actornet, exploration_constant, epsilon, epsilon_decay)

            visualization_states = [self.game.get_state()]
            
            while not self.game.end_state():
                searches = int((final_search_games - initial_search_games) / (actual_games) * g_a + initial_search_games)
                tree.search(searches) # The search games increases linearly from initial_search_games to final_search_games for every actual game
                distribution = tree.get_distribution()
                replay_buffer.append((self.game.get_player() + tree.root.state, distribution))

                action = np.argmax(distribution)
                self.game.do_action(action)
                visualization_states.append(self.game.get_state())
                tree.do_action(action)

            replay_probabilites = np.arange(1, len(replay_buffer) + 1) # Make it so cases later in the replay buffer are more likely to be selected
            replay_probabilites = replay_probabilites / np.sum(replay_probabilites)
            batch_indexes = np.random.choice(len(replay_buffer), size=np.minimum(len(replay_buffer), 50), replace=False, p=replay_probabilites)
            self.actornet.train(replay_buffer, batch_indexes)
        
            # Save the actor network at regular intervals
            if (g_a + 1) % save_interval == 0:
                self.actornet.save(g_a + 1)
            
            # Play out a game if the conditions are true
            # Save every board state in visualization_states for the visualization
            if visualize and (g_a + 1) % visualization_frequency == 0:
                self.game.visualize(visualization_frame_delay, visualization_states)
