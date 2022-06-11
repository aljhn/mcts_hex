import numpy as np
from collections import defaultdict


# Class for making the search-tree easier to deal with
class Node:

    def __init__(self, state, player, parent=None, parent_action=None):
        self.state = state
        self.player = player
        self.children = {}
        self.parent = parent
        self.parent_action = parent_action


class MonteCarloTree:

    def __init__(self, root_state, root_player, game, actornet, exploration_constant, epsilon, epsilon_decay):
        self.root = Node(root_state, root_player)
        self.game = game
        self.actornet = actornet
        self.exploration_constant = exploration_constant
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.N = defaultdict(lambda: 0)
        self.Q = defaultdict(lambda: 0)
    

    # Compute the tree-policy using the current Q values together with the UCT for the exploration bonus
    def policy(self, state, action, player, c=1):
        if player == 2:
            c *= -1
        return self.Q[(state, action)] + c * np.sqrt(np.log(self.N[state]) / (1 + self.N[(state, action)]))
    

    # Perform multiple Monte Carlo Tree Searches
    def search(self, searches):
        for s in range(searches):
            
            node = self.root
            self.game.initialize(self.root.state, self.root.player)

            # Search for a leaf node with the tree policy
            # Move from the root node to the first child node found using the given policy
            # Player 1 wants to maximize the policy while player 2 wants to minimize it
            while node.children: # while the node has children
                actions = list(node.children.keys())
                best_action = actions[0]
                best_policy = policy = self.policy(node.state, best_action, self.game.player, self.exploration_constant)
                for i in range(1, len(actions)): # All actions should be valid
                    policy = self.policy(node.state, actions[i], self.game.player, self.exploration_constant)
                    if (self.game.player == 1 and policy > best_policy) or (self.game.player == 2 and policy < best_policy):
                        best_action = actions[i]
                        best_policy = policy
                
                self.game.do_action(best_action)
                node = node.children[best_action]

            # Expand leaf node
            # Only expand nodes to valid actions, so only if the game has not finished
            # Every possible child node is added, so the tree grows rapidly in the beginning
            # Do this by doing the action in the actual game, creating the node object and then initializing to before the node object creation
            # Then randomly select one of the child nodes to move there
            if not self.game.end_state():
                actions = self.game.get_actions()
                valid_actions = []
                for i in range(len(actions)):
                    if actions[i] == 1:
                        valid_actions.append(i)

                        self.game.do_action(i)
                        child = Node(self.game.get_state(), self.game.player, node, i)
                        node.children[i] = child

                        self.game.initialize(node.state, node.player)
                
                action = np.random.choice(valid_actions)
                self.game.do_action(action)
                node = node.children[action]
            
            # Rollout simulations
            # Use epsilon-greediness, meaning a random valid action is chosen by random chance
            # Otherwise choose actions based on the current actor network
            while not self.game.end_state():
                if np.random.rand() < self.epsilon:
                    actions = self.game.get_actions()
                    valid_actions = []
                    for i in range(len(actions)):
                        if actions[i] == 1:
                            valid_actions.append(i)
                    action = np.random.choice(valid_actions)
                else:
                    action = self.predict()
                self.game.do_action(action)
            
            # Backpropagate from the node from before the rollouts began to the root
            # Get the result from the game and use this to update the Q values
            # The N values are incremented for each visited node
            result = self.game.get_result()
            z = 1 if result == 1 else -1
            while node.parent is not None:
                state = node.parent.state
                action = node.parent_action
                self.N[state] += 1
                self.N[(state, action)] += 1
                self.Q[(state, action)] = ((self.N[(state, action)] - 1) * self.Q[(state, action)] + z) / self.N[(state, action)]
                node = node.parent
        
        self.epsilon *= self.epsilon_decay # Decrease epsilon
    

    # Do a prediction with the actor network to get a distribution over moves from the current state
    # Remove invalid actions by multiplying with the valid actions
    # Return the argmax as the action
    def predict(self):
        actions = self.game.get_actions()
        distribution = self.actornet.predict(self.game.get_player() + self.game.get_state())
        distribution *= actions # Elementwise multiplication
        #distribution /= np.sum(distribution)
        action = np.argmax(distribution)
        return action


    # Get the distribution over visted child nodes from the root
    # Only based on the visits and has nothing to do with the actor network
    # As the whole distribution and not a single action is returned it is necessary to normalize it
    def get_distribution(self):
        distribution = np.zeros_like(self.game.get_actions())
        for action in self.root.children: # Should always be valid actions
            distribution[action] = self.N[(self.root.state, action)]
        distribution /= np.sum(distribution)
        return distribution
    

    # Move the root of the tree to the children at the chosen action
    # Remove the new roots parent and parent_action to make the while loops stop while searching
    def do_action(self, action):
        self.root = self.root.children[action]
        self.root.parent = None
        self.root.parent_action = None
