import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Hex:
    
    def __init__(self, board_size):
        self.board_size = board_size
    
        self.initialize()


    # Used to convert between index representations
    def to_vector_index(self, i, j):
        return i * self.board_size + j
    

    # Used to convert between index representations
    def from_vector_index(self, index):
        i = index // self.board_size
        j = index % self.board_size
        return (i, j)
    
    
    # Store an adjacency graph describing connections on the board
    # Update the graph by checking every cell in the 6-neighborhood defined in the doc
    # Function is called whenever an action is made
    def update_graph(self, action, player):
        graph_index = int(player - 1)

        i, j = self.from_vector_index(action)

        neighbor_index = self.to_vector_index(i - 1, j)
        if i - 1 >= 0 and self.board[neighbor_index] == player:
            self.graph[graph_index, action, neighbor_index] = 1
            self.graph[graph_index, neighbor_index, action] = 1

        neighbor_index = self.to_vector_index(i, j - 1)
        if j - 1 >= 0 and self.board[neighbor_index] == player:
            self.graph[graph_index, action, neighbor_index] = 1
            self.graph[graph_index, neighbor_index, action] = 1

        neighbor_index = self.to_vector_index(i + 1, j)
        if i + 1 < self.board_size and self.board[self.to_vector_index(i + 1, j)] == player:
            self.graph[graph_index, action, neighbor_index] = 1
            self.graph[graph_index, neighbor_index, action] = 1

        neighbor_index = self.to_vector_index(i, j + 1)
        if j + 1 < self.board_size and self.board[self.to_vector_index(i, j + 1)] == player:
            self.graph[graph_index, action, neighbor_index] = 1
            self.graph[graph_index, neighbor_index, action] = 1

        neighbor_index = self.to_vector_index(i - 1, j + 1)
        if i - 1 >= 0 and j + 1 < self.board_size and self.board[self.to_vector_index(i - 1, j + 1)] == player:
            self.graph[graph_index, action, neighbor_index] = 1
            self.graph[graph_index, neighbor_index, action] = 1

        neighbor_index = self.to_vector_index(i + 1, j - 1)
        if i + 1 < self.board_size and j - 1 >= 0 and self.board[self.to_vector_index(i + 1, j - 1)] == player:
            self.graph[graph_index, action, neighbor_index] = 1
            self.graph[graph_index, neighbor_index, action] = 1
    

    # Used for early debugging
    def print_state(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                value = int(self.board[self.to_vector_index(i, j)])
                print(value, end=" ")
            print()
        print()


    # Reset the board state and whose turn it is
    # If the state and player parameters are not None, set the board to that
    def initialize(self, state=None, player=1):
        self.board = np.zeros(self.board_size * self.board_size)
        self.actions = np.ones(self.board_size * self.board_size)
        self.player = player
        self.result = 0

        self.graph = np.zeros((2, self.board_size * self.board_size, self.board_size * self.board_size))

        if state is not None:
            for i in range(0, len(state), 2):
                index = i // 2
                if state[i] == 1:
                    self.board[index] = 1
                    self.update_graph(index, 1)
                    self.actions[index] = 0
                elif state[i + 1] == 1:
                    self.board[index] = 2
                    self.update_graph(index, 2)
                    self.actions[index] = 0


    # Board is stored and used as a 1-D list inside this class, where elements are either 0, 1 or 2
    # For external use, convert the board to a tuple with with twice the length, where every value is expanded into two
    # 0 -> 0, 0
    # 1 -> 0, 1
    # 2 -> 1, 0
    def get_state(self):
        state = [0 for i in range(2 * len(self.board))]
        for i in range(len(self.board)):
            if self.board[i] == 1:
                state[2 * i] = 1
            elif self.board[i] == 2:
                state[2 * i + 1] = 1
        return tuple(state)
    

    # Do the same with the player
    def get_player(self):
        player = [0, 0]
        player[self.player - 1] = 1
        return tuple(player)


    # Return the list of valid actions
    # These are updated whenever an action is made
    def get_actions(self):
        return self.actions


    # Do the given action
    # Actions are represented by an index in action-space
    # Keep internal track of whose turn it is, and set the board to that player
    # Then remove that action from the valid actions and update the player
    def do_action(self, action):
        self.board[action] = self.player
        self.update_graph(action, self.player)
        self.actions[action] = 0

        if self.player == 1:
            self.player = 2
        else:
            self.player = 1


    # Check if the game is finished by performing multiple depth-first searches
    # Player 1 wins if there is a path spanning all the rows, so do one DFS from every cell on the first row
    # If the DFS finds a cell on the final row, set the result and return True
    # Do the same for player 2, except spanning every column
    # Return False if no paths were found for any players
    # The DFSes use the adjacency graph which is updated with each move
    def end_state(self):
        for i in range(self.board_size):
            start_index = self.to_vector_index(0, i) # Start at every node on the top row
            stack = [start_index]
            visited = np.zeros(self.board_size * self.board_size) # Keep track of visited cells for the DFS
            visited[start_index] = 1 # Set the first cell to visited
            while len(stack) > 0:
                cell = stack.pop()
                for j in range(self.board_size * self.board_size): # Iterate over every cell
                    if visited[j] == 0 and self.graph[0, cell, j] == 1: # Graph has 2 2D-lists, one for each player
                        x, y = self.from_vector_index(j)
                        if x == self.board_size - 1: # A node has been found on the final row beginning from the first row
                            self.result = 1
                            return True
                        stack.append(j)
                        visited[j] = 1
                
        for i in range(self.board_size): # Do the same thing for player 2, but with the adjacency graph of player 2 and columns
            start_index = self.to_vector_index(i, 0)
            stack = [start_index]
            visited = np.zeros(self.board_size * self.board_size)
            visited[start_index] = 1
            while len(stack) > 0:
                cell = stack.pop()
                for j in range(self.board_size * self.board_size):
                    if visited[j] == 0 and self.graph[1, cell, j] == 1:
                        x, y = self.from_vector_index(j)
                        if y == self.board_size - 1:
                            self.result = 2
                            return True
                        stack.append(j)
                        visited[j] = 1

        return False
    

    # Result is set whenever the end state is found
    def get_result(self):
        return self.result


    # Draw a complete game with networkx and matplotlib
    # For every state, create a networkx graph
    # Then add every edge in the game, meaning every cell that is connected to eachother
    # Transform thei position of every cell / node to screen space to make the figure look like it should
    # Open cells are grey and pegged cells are blue
    def visualize(self, frame_delay, states):
        plt.ion()  # Needed to allow matplotlib to override the current figure

        for s in range(len(states)):
            state = states[s]
            G = nx.Graph()

            node_colors = []
            node_positions = {}

            already_drawn = {}
            for x in range(0, len(state), 2): # Loop through every cell
                index = x // 2
                if state[x] == 1:
                    node_colors.append("red")
                elif state[x + 1] == 1:
                    node_colors.append("black")
                else:
                    node_colors.append("white")
                    
                i, j = self.from_vector_index(index)

                # Coordinate transformation to make the graph a diamond
                # The diamond is shaped like to equilateral triangles stacked on top of eachother
                # Do a transformation composed of a rotation and a scaling
                # Could be done with matrix multiplications, but the end result is shown here simplified
                node_x = j / 1.732 - i / 1.732  # sqrt(3) = 1.732
                node_y = - j - i
                node_positions[index] = [node_x, node_y]

                # Find every cell connected to the current one, both open and pegged cells
                neighbors = []
                if i - 1 >= 0:
                    neighbors.append(self.to_vector_index(i - 1, j))
                if j - 1 >= 0:
                    neighbors.append(self.to_vector_index(i, j - 1))
                if i - 1 >= 0 and j + 1 < self.board_size:
                    neighbors.append(self.to_vector_index(i - 1, j + 1))
                if i + 1 < self.board_size:
                    neighbors.append(self.to_vector_index(i + 1, j))
                if j + 1 < self.board_size:
                    neighbors.append(self.to_vector_index(i, j + 1))
                if i + 1 < self.board_size and j - 1 >= 0:
                    neighbors.append(self.to_vector_index(i + 1, j - 1))

                # Loop through these neighbors
                # If the edge is not already drawn add it to the networkx graph
                for y in neighbors:
                    if (index, y) not in already_drawn:
                        G.add_edge(index, y)
                        already_drawn[(index, y)] = True

            # Make the colors correspond to the nodes in the graph
            # Basically a rearranging of the list
            node_color_map = [node_colors[node] for node in G.nodes()]

            nx.draw(G, pos=node_positions, node_color=node_color_map, font_color="black")
            plt.show()
            plt.pause(frame_delay)
            
            if s < len(states) - 1:
                plt.clf()  # Clear frame

        plt.ioff()
