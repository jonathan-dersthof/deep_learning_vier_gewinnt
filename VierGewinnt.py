import numpy
from Agent import *

class VierGewinnt:
    def __init__(self):
        self.board = numpy.zeros((6, 7))
        self.current_player = random.choice([-1, 1])
        self.done = False
        self.outcome = None
        self.players = {
            1 : "Spieler 1",
            -1 : "Spieler 2"
        }

    def get_state(self) -> numpy.ndarray:
        return self.board.copy()

    def get_state_str(self):
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

    def get_valid_moves(self):
        return [c for c in range(7) if self.board[0, c] == 0]

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

    def random_move(self):
        valid_moves = self.get_valid_moves()
        action = valid_moves[numpy.random.randint(0, len(valid_moves))]

        return action

    def make_move(self, col : int):
        for row in list(reversed(range(6))):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                return [row, col]

        return None

    def step(self, action):
        move = self.make_move(action)
        won = self.check_win(move)

        if won:
            self.outcome = "win"
            self.done = True

        elif self.check_draw():
            self.outcome = "draw"
            self.done = True

    def show_board(self):
        print(self.get_state_str())
