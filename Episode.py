from Agent import Agent
from VierGewinnt import VierGewinnt
from numpy import random

class Episode:
    def __init__(self,
                 agent_a,
                 agent_b = None):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.game_states_str = []

        self.env = VierGewinnt()

    def run_action(self, env):
        if env.current_player == 1:
            move = env.model_move(self.agent_a)
        elif self.agent_b:
            move = env.model_move(self.agent_b)
        else:
            move = env.random_move()

        return move[1]

    def final_rewards(self):
        if self.env.check_draw():
            if self.agent_a.memory:
                last = list(self.agent_a.memory[-1])
                last[2] = self.agent_a.draw_reward
                last[4] = True
                self.agent_a.memory[-1] = tuple(last)
                self.agent_a.reward += self.agent_a.draw_reward

            if self.agent_b and self.agent_b.memory:
                last = list(self.agent_b.memory[-1])
                last[2] = self.agent_b.draw_reward
                last[4] = True
                self.agent_b.memory[-1] = tuple(last)
                self.agent_b.reward += self.agent_b.draw_reward
        else:
            if self.env.current_player == 1:
                if self.agent_a.memory:
                    last_memory = list(self.agent_a.memory[-1])
                    last_memory[2] = self.agent_a.lose_reward
                    last_memory[4] = True
                    self.agent_a.memory[-1] = tuple(last_memory)
                    self.agent_a.reward += last_memory[2]

            elif self.env.current_player == -1 and self.agent_b:
                if self.agent_b.memory:
                    last_memory = list(self.agent_b.memory[-1])
                    last_memory[2] = self.agent_b.lose_reward
                    last_memory[4] = True
                    self.agent_b.memory[-1] = tuple(last_memory)
                    self.agent_b.reward += last_memory[2]

    def run(self):
        self.game_states_str = self.env.get_board()

        self.agent_a.reward = 0
        if self.agent_b:
            self.agent_b.reward = 0

        moves = 0

        while not self.env.done:
            self.run_action(self.env)
            self.game_states_str.append(self.env.get_board())

            self.env.check_draw()
            self.env.current_player *= -1
            moves += 1


        self.final_rewards()
