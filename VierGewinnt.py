import numpy
from Agent import *

class VierGewinnt:
    def __init__(self):
        self.board = numpy.zeros((6, 7))
        self.current_player = random.choice([-1, 1])
        self.done = False

    def get_state(self) -> numpy.ndarray:
        return self.board.copy()

    def get_valid_moves(self):
        return [c for c in range(7) if self.board[0, c] == 0]

    def make_move(self, col : int):
        for row in list(reversed(range(6))):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                return [row, col]

        return None

    def random_move(self):
        state = self.board.copy()

        valid_moves = [col for col in range(7) if state[0, col] == 0]
        random_col = valid_moves[numpy.random.randint(0, len(valid_moves))]
        move = self.make_move(random_col)

        if self.check_win(move):
            self.done = True

        return move


    def model_move(self, agent : Agent):
        state = self.board.copy()

        action = agent.act(self)
        move = self.make_move(action)

        if self.check_win(move):
            reward = agent.winner_reward
            self.done = True
        else:
            reward = agent.survive_reward

        next_state = self.board.copy()

        if self.current_player == -1:
            next_state *= -1
            state *= -1

        agent.remember(state, action, reward, next_state, self.done)

        if reward == agent.winner_reward:
            agent.reward = agent.reward/2 + agent.winner_reward
        else:
            agent.reward += reward

        return move

    def check_win(self, tile) -> bool:
        row = tile[0]
        col = tile[1]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            for delta in [1, -1]:
                r, c = row + dr * delta, col + dc * delta

                while 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == self.current_player:
                    count += 1
                    r += dr * delta
                    c += dc * delta

            if count >= 4:
                return True

        return False

    def check_draw(self):
        if not 0 in self.get_state():
            self.done = True
            return True
        else:
            return False

    def show_board(self):
        print(self.get_board())

    def get_board(self):
        players = {
            numpy.float64(1.0): " X ",
            numpy.float64(-1.0): " O ",
            numpy.float64(0.0): " . "
        }
        board_str = ""

        for row in range(6):
            row_str = ""

            for col in range(7):
                row_str += players[self.board[row][col]]
            board_str += f"{6 - row} {row_str}\n"

        board_str += f"   1  2  3  4  5  6  7\n"

        return board_str
