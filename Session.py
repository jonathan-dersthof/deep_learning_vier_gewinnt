from VierGewinnt import VierGewinnt
from Agent import Agent
from logic import *


class Session:
    def __init__(self,
                 directory,
                 agent_a,
                 agent_b = None
                 ):
        self.agent_a = agent_a

        if agent_b:
            self.agent_b = agent_b

        self.directory = directory
        self.log_data = []

        print(self.directory)

    def save_episode(self,
                     episode,
                     episodes,
                     episode_total_reward,
                     step):
            if episode % step == 0:
                print(f"Episode: {episode}/{episodes}, Score: {episode_total_reward}, Epsilon: {self.agent_a.epsilon:.2f}")
                self.agent_a.save_model(f"training/{self.directory}/models/agent_episode{episode}.pth")

    def run_action(self, env, player):
        if player == 1:
            move = env.model_move(self.agent_a, player)
            action = move[0]
        elif self.agent_b:
            move = env.model_move(self.agent_b, player)
            action = move[0]
        else:
            move = env.random_move(self.agent_a, player)
            action = move[0]

        return action

    def run_episode(self, total_episodes, current_episode):
        env = VierGewinnt()
        state = env.board.copy()

        self.agent_a.reward = 0

        if self.agent_b:
            self.agent_b.reward = 0

        moves = 0

        player = 1
        action = None

        while env.running:
            if env.check_draw():
                self.agent_a.reward += self.agent_a.draw_reward
                self.agent_a.remember(state, action, self.agent_a.draw_reward, env.board.copy(), True)

                if self.agent_b:
                    self.agent_b.reward += self.agent_b.draw_reward
                    self.agent_b.remember(state, action, self.agent_b.draw_reward, env.board.copy(), True)

            elif player == 1:
                move = env.model_move(self.agent_a, player)
                action = move[0]
                state = env.board.copy()

            else:
                move = env.random_move(self.agent_a, player)
                action = move[0]
                state = env.board.copy()

            moves += 1
            player *= -1

        self.agent_a.replay()
        self.log_data.append([current_episode, self.agent_a.reward, self.agent_a.epsilon])

        if current_episode < 1000:
            self.save_episode(current_episode, total_episodes, self.agent_a.reward, 100)
        else:
            self.save_episode(current_episode, total_episodes, self.agent_a.reward, 1000)

    def run(self, episodes):
        for episode in range(episodes):
            self.run_episode(episodes, episode)
