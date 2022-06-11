import numpy as np


class Tournament:

    def __init__(self, agents, game):
        self.agents = agents
        self.M = len(agents)
        self.game = game

    
    # Play a game between two agents
    # Return 1 if the first agent won and -1 if the second agent won
    # Also return every state that the game went through for the purpose of visualization later
    def play(self, agents, start_player):
        self.game.initialize(player=start_player)
        states = [self.game.get_state()]
        while not self.game.end_state():
            agent = self.agents[agents[self.game.player]]
            action = agent.get_action(self.game.get_player() + self.game.get_state(), self.game.get_actions())
            self.game.do_action(action)
            states.append(self.game.get_state())
        result = self.game.get_result()
        return result, states


    # Play every agent against every other agent
    # Play a set amount of games from every possible combination of which agent is which player and who begins
    # Do twice the amount of games described in the assignment, to make sure that both agents play as both players
    def run(self, games):
        results = np.zeros((self.M, self.M), dtype=int)
        for i in range(self.M):
            for j in range(self.M):
                if i == j: # Dont play against itself
                    continue
                for g in range(games):
                    start_player = 1 if g % 2 == 0 else 2 # Alternate who starts
                    agents = {1: i, 2: j} # Player 1 is set to the agent at index i, and player to index j
                    result, states = self.play(agents, start_player)
                    if result == 2:
                        result = -1
                    results[i, j] += result
        return results
    

    # Visualize games through an interactive interface
    # Select two agents, choose who starts and play the game
    def visualize(self, frame_delay):
        while True:
            try:
                play = input("Visualize a game [y/N]: ")
                if play != "y":
                    break
                p1 = int(input("Player 1 index: "))
                p2 = int(input("Player 2 index: "))
                start_player = int(input("Start player [1/2]: "))
                agents = {1: p1, 2: p2}
                result, states = self.play(agents, start_player)
                self.game.visualize(frame_delay, states)
                if result == 1:
                    print("Winner: Player 1\n")
                else:
                    print("Winner: Player 2\n")
            except:
                print("Error")
                continue
