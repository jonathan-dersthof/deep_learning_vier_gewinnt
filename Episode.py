from Agent import Agent
from VierGewinnt import VierGewinnt
from numpy import random

class Episode:
    def __init__(self,
                 env : VierGewinnt,
                 agent_a : Agent,
                 agent_b : Agent = None):
        self.env = env

        self.agent_a = agent_a
        self.agent_b = agent_b

        self.game_states_str = []

    def final_rewards(self):
        match self.env.outcome:
            case "draw":
                self.agent_a.correct_last_reward(self.agent_a.draw_reward)

                if self.agent_b:
                    self.agent_b.correct_last_reward(self.agent_b.draw_reward)
            case "win":
                match self.env.players[self.env.current_player]:
                    case "Spieler 1":
                        self.agent_a.correct_last_reward(self.agent_a.win_reward)

                        if self.agent_b:
                            self.agent_b.correct_last_reward(self.agent_b.lose_reward)

                    case "Spieler 2":
                        self.agent_a.correct_last_reward(self.agent_a.lose_reward)

                        if self.agent_b:
                            self.agent_b.correct_last_reward(self.agent_b.win_reward)

    def run_action(self):
        state = self.env.get_state()

        if self.env.current_player == 1:
            action = self.agent_a.act(self.env)
        elif self.agent_b:
            action = self.agent_b.act(self.env)
        else:
            action = self.env.random_move()

        self.env.step(action)
        next_state = self.env.get_state()

        if self.env.current_player == -1:
            next_state *= -1
            state *= -1

        if self.env.current_player == 1:
            reward = self.agent_a.survive_reward
            self.agent_a.reward += reward

            self.agent_a.remember(state, action, reward, next_state, self.env.done)
        elif self.agent_b:
            reward = self.agent_b.survive_reward
            self.agent_b.reward += reward

            self.agent_b.remember(state, action, reward, next_state, self.env.done)

    def run(self):
        self.game_states_str.append(self.env.get_state_str())

        self.agent_a.reward = 0
        if self.agent_b:
            self.agent_b.reward = 0

        while not self.env.done:
            self.run_action()
            self.game_states_str.append(self.env.get_state_str())

            if not self.env.done:
                self.env.current_player *= -1

        self.final_rewards()
