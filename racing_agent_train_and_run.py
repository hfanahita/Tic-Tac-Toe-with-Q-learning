import numpy as np
import random
import pickle
class Board:
    def __init__(self):
        self.state = [0] * 25

    def show_state(self, state):
      row = ''
      for i in range(1, 26):
          if i % 5 == 1:
              row = '| '
          if state[i - 1] == 1:
              row += 'X | '
          elif state[i - 1] == -1:
              row += 'O | '
          else:
              row += '  | '
          if i % 5 == 0:
              print(row)


    def get_state(self):
        return self.state

    def translate_move(self, move):
        action = move[0]*5 + move[1]
        return action

    def update_state(self, player, action):
        """
        player = 1 or -1,
        action = index of the cell of the board that is about to be played
        """
        if self.state[action] == 0:
          self.state[action] = player

    @staticmethod
    def full_board(state):
        if not (0 in state):
            return True
        return False


class Agent:
    def __init__(self, player):
        self.player = player
        self.exploration_rate = 0.1
        self.learning_rate = 0.4
        self.gamma = 0.3
        self.possible_moves = list(range(0, 25))
        self.board = Board()
        self.q_table = {}
        self.epsilon_decay = 0.995


    def get_q_value(self, state, action):
        state_id = self.state_id(state)
        value = self.q_table.get((state_id, action), 0)
        if value == 0:
            value = self.estimate_value(state, action)
            self.update_q_value(state_id, action, value)
        return value

    def estimate_value(self, state, action):
        temp_state = list(state)  # Copy state
        temp_state[action] = self.player  # Simulate taking the action

        row, col = divmod(action, 5)
        opponent = -self.player

        # Calculate how close we are to winning in rows, columns, and diagonals
        row_sum = sum(temp_state[row * 5:(row + 1) * 5])
        col_sum = sum(temp_state[col:25:5])
        diag1_sum = sum(temp_state[0:25:6]) if row == col else 0
        diag2_sum = sum(temp_state[4:21:4]) if row + col == 4 else 0

        # Calculate opponent's winning chances that we are blocking
        row_block = sum(1 for i in range(5) if temp_state[row * 5 + i] == opponent)
        col_block = sum(1 for i in range(5) if temp_state[col + i * 5] == opponent)
        diag1_block = sum(1 for i in range(5) if temp_state[i * 6] == opponent) if row == col else 0
        diag2_block = sum(1 for i in range(5) if temp_state[4 + i * 4] == opponent) if row + col == 4 else 0

        # Assign heuristic Q-value
        q_value = (
                3 * (row_sum + col_sum + diag1_sum + diag2_sum)  # Reward progress toward winning
                + 5 * (row_block + col_block + diag1_block + diag2_block)  # Reward blocking opponent
        )

        # Add a small exploration bonus if no major heuristic applies
        if q_value == 0:
            q_value = 0.01

        return q_value

    def update_q_value(self, state_id, action, new_value):
        self.q_table[(state_id, action)] = new_value

    def move(self, cell):
        self.board.update_state(self.player, cell)
        self.possible_moves.remove(cell)

    def find_high_priority_move(self, state):
        """
        Finds a move that leads to a guaranteed win within two moves.
        Returns the best move if found, otherwise returns None.
        """
        available_moves = self.find_available_moves(state)

        for move in available_moves:
            temp_state = state.copy()
            temp_state[move] = self.player  # Simulate current player's move

            # Check if this move leads to a forced win next turn
            if self.can_win_next_turn(temp_state):
                return move  # High-priority move found

        return None  # No forced two-move win found

    def can_win_next_turn(self, state):
        """
        Checks if the opponent has no way to stop a win in the next move.
        """
        opponent = -self.player
        available_moves = self.find_available_moves(state)

        for move in available_moves:
            temp_state = state.copy()
            temp_state[move] = self.player  # Simulate next move

            if self.win(temp_state):  # If this leads to a win, return True
                return True

        return False
    def find_winning_move(self, state):
        for move in self.find_available_moves(state):
            temp_state = state.copy()
            temp_state[move] = self.player
            if self.win(temp_state):
                return move
        return None

    def state_id(self, state):
        return " ".join([str(cell) for cell in state])

    def find_blocking_move(self, state):
        """Finds a move that prevents the opponent from winning"""
        for move in self.find_available_moves(state):
            temp_state = state.copy()
            temp_state[move] = -self.player
            if self.game_over(temp_state):
                return move
        return None

    def opponent_moves(self, state):
        for move in self.find_available_moves(state):
            new_state = state.copy()
            new_state[move] = -self.player
            if self.game_over(new_state):
                return True
        return False

    def game_over(self, state):
        opponent = self.player * -1
        # checking rows
        for i in range(0, 25, 5):
            if sum(state[i:i + 5]) == opponent * 5:
                return True
        # checking columns
        for i in range(0, 5):
            if sum(state[i:25:5]) == opponent * 5:
                return True
        # checking diagonals
        if sum(state[0:25:6]) == opponent * 5:
            return True
        if sum(state[4:21:4]) == opponent * 5:
            return True
        return False

    def win(self, state):
        # checking rows
        for i in range(0, 25, 5):
            if sum(state[i:i + 5]) == self.player * 5:
                return True
        # checking columns
        for i in range(0, 5):
            if sum(state[i:25:5]) == self.player * 5:
                return True
        # checking for diagonals
        if sum(state[0:25:6]) == self.player * 5:
            return True
        if sum(state[4:21:4]) == self.player * 5:
            return True
        return False

    def tie(self, state):
        if Board.full_board(state):
            if not self.win(state) and not self.game_over(state):
                return True
            return False
        return False

    def strategic_exploration(self, state):
        """Encourages moves that build towards a win by checking rows, columns, and diagonals."""
        available_moves = self.find_available_moves(state)
        move_scores = {}

        for move in available_moves:
            temp_state = state.copy()
            temp_state[move] = self.player  # Simulate making this move

            # Count how many pieces are in the same row/column/diagonal
            row, col = divmod(move, 5)
            row_score = sum(temp_state[row * 5:(row + 1) * 5])  # Row sum
            col_score = sum(temp_state[col:25:5])  # Column sum
            diag1_score = sum(temp_state[0:25:6]) if row == col else 0  # Main diagonal
            diag2_score = sum(temp_state[4:21:4]) if row + col == 4 else 0  # Anti-diagonal

            total_score = row_score + col_score + diag1_score + diag2_score
            move_scores[move] = total_score

        # Pick the move with the highest score
        return max(move_scores, key=move_scores.get)

    def reward(self, state):
        if self.win(state):
            return 20
        if self.game_over(state):
            return -10 - (25 - state.count(0))
        if self.tie(state):
            return -40  # Increase penalty for ties
        if self.opponent_moves(state):
            return -10
        return -3

    def update_q_table(self, current_state, move, reward, next_state):
        current_state_id = self.state_id(current_state)
        current_q_value = self.get_q_value(current_state, move)

        if self.is_terminal(next_state):  # Assuming is_terminal checks for win/loss/draw
            max_next_q_value = 0

        else:
            best_next_action, max_next_q_value = self.max_q_value(next_state)
            if best_next_action is None:
                max_next_q_value = 0

        updated_q_value = current_q_value + self.learning_rate * (
                reward + self.gamma * max_next_q_value - current_q_value)

        self.update_q_value(current_state_id, move, updated_q_value)

    def is_terminal(self, state):
        return self.win(state) or self.tie(state) or self.game_over(state)

    def max_q_value(self, state):
        state_id = self.state_id(state)
        available_moves = self.find_available_moves(state)
        if not available_moves:
            return None, 0  # No move available

        move_q_values = {move: self.q_table.get((state_id, move), 0) for move in available_moves}
        best_move = max(move_q_values, key=move_q_values.get)

        return best_move, move_q_values[best_move]

    def find_available_moves(self, state):
        return [index for index, value in enumerate(state) if value == 0]

    def first_move(self, state):
        return np.count_nonzero(state) == 0

    def play(self):
        state = self.board.get_state()
        state_id = self.state_id(self.board.get_state())

        # Check if the game is over
        if self.is_terminal(state):
            return None
        if self.first_move(state):
            action = random.choice([7, 12, 11, 13, 17])
            self.move(action)
            new_state = self.board.get_state()
            reward = self.reward(new_state)  # Compute reward for winning
            new_state_id = self.state_id(new_state)
            self.update_q_table(state, action, reward, new_state)
            return action
        # Check for a winning move
        winning_move = self.find_winning_move(state)
        if winning_move is not None:
            self.move(winning_move)
            action = winning_move
            new_state = self.board.get_state()
            reward = self.reward(new_state)  # Compute reward for winning
            new_state_id = self.state_id(new_state)
            self.update_q_table(state, winning_move, reward, new_state)
            return action  # Immediate win
        blocking_move = self.find_blocking_move(state)
        # Check for a blocking move
        if blocking_move is not None:
            self.move(blocking_move)
            action = blocking_move
            new_state = self.board.get_state()
            reward = self.reward(new_state)  # Compute reward for blocking
            self.update_q_table(state, blocking_move, reward, new_state)
            return action  # Block opponent immediately
        high_priority_move = self.find_high_priority_move(state)
        if high_priority_move is not None:
            self.move(high_priority_move)
            new_state = self.board.get_state()
            reward = self.reward(new_state)
            self.update_q_table(state, high_priority_move, reward, new_state)
            return high_priority_move
        # Decide whether to explore or exploit
        if np.random.uniform(0, 1) <= self.exploration_rate:
            # Exploration: Pick a random move
            random_move = self.strategic_exploration(state)
            self.move(random_move)
            action = random_move
        else:
            # Exploitation: Pick the best Q-value move
            original_move = self.max_q_value(state)[0]
            self.move(original_move)
            action = original_move

        # Update the Q-table
        new_state = self.board.get_state()
        new_state_id = self.state_id(new_state)
        reward = self.reward(new_state)
        self.update_q_table(state, action, reward, new_state)

        return action


with open("q_table.pkl", "rb") as f:
    data = pickle.load(f)

turn = int(input())
player = 1
agent = Agent(player)
agent.q_table = data
state = [0]*25
def output_action(action):
    column = action % 5
    row = action//5
    return [row, column]
first = True
while not agent.board.full_board(state):
    if turn == player:
        # agent turn
        action = agent.play()
        output = output_action(action)
        print(output[0], "-", output[1])
    else:
        turn = -1
        action = input().split("-")
        row = int(action[0])
        column = int(action[1])
        move = row * 5 + column
        agent.board.update_state(-player, move)
        agent.possible_moves.remove(move)

    state = agent.board.get_state()
    turn = -1*turn
