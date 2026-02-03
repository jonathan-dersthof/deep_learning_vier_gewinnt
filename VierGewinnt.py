import numpy


class VierGewinnt:
    def __init__(self):
        self.board = numpy.zeros((6, 7))
        self.running = True

    def get_state(self) -> numpy.ndarray:
        return self.board

    def make_move(self, col : int, player : int):
        for row in list(reversed(range(6))):
            if self.board[row, col] == 0:
                self.board[row, col] = player
                return [row, col]

        return False

    @staticmethod
    def in_bounds(tile : list[int]) -> bool:
        return 0 <= tile[0] < 6 and 0 <= tile[1] < 7

    def random_move(self, opponent, player):
        state = self.board.copy()

        valid_moves = [col for col in range(7) if state[0, col] == 0]
        random_col = valid_moves[numpy.random.randint(0, len(valid_moves))]
        move = self.make_move(random_col, -1)

        if self.check_win(move, player):
            reward = opponent.winner_reward
            self.running = False
            opponent.remember(state, random_col, reward, self.board.copy(), self.running)

        return move

    def model_move(self, agent, player):
        state = self.board.copy()

        if player == -1:
            state *= -1

        action = agent.act(state, self)
        move = self.make_move(action, player)

        if self.check_win(move, player):
            reward = agent.winner_reward
            self.running = False
        else:
            reward = agent.survive_reward

        next_state = self.board.copy()

        if player == -1:
            next_state *= -1

        agent.remember(state, action, reward, next_state, self.running)
        agent.reward += reward

        return move

    def check_win(self, tile, player) -> bool:
        row = tile[0]
        col = tile[1]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            for delta in [1, -1]:
                r, c = row + dr * delta, col + dc * delta

                while 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == player:
                    count += 1
                    r += dr * delta
                    c += dc * delta

            if count >= 4:
                return True

        return False

    def check_draw(self):
        if not 0 in self.get_state():
            self.running = False
            return True
        else:
            return False

    def show_board(self):
        players = {
            numpy.float64(1.0) : " X ",
            numpy.float64(-1.0) : " O ",
            numpy.float64(0.0): " . "
        }
        board_str = ""

        for row in range(6):
            row_str = ""

            for col in range(7):
                row_str += players[self.board[row][col]]
            board_str += f"{6 - row} {row_str}\n"

        board_str += f"   1  2  3  4  5  6  7\n"
        print(board_str)
