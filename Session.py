import pandas

from VierGewinnt import VierGewinnt
from Agent import Agent
from logic import *


class Session:
    def __init__(self,
                 directory,
                 episodes,
                 agent_a,
                 agent_b = None
                 ):
        self.agent_a = agent_a
        self.agent_b = agent_b

        self.directory = directory

        self.log_data_a = []
        self.log_data_b = []

        self.current_episode = 0
        self.total_episodes = episodes

    def save_weights(self,
                     agent,
                     a_or_b,
                     step,
                     ):
        current_episode = self.current_episode
        total_episodes = self.total_episodes

        if self.current_episode % step == 0:
            print(f"Episode: {current_episode}/{total_episodes}, Score: {agent.reward}, Epsilon: {agent.epsilon:.2f}")
            if self.agent_b:
                agent.save_model(f"training/{self.directory}/models/agent_{a_or_b}_episode{current_episode}.pth")
            else:
                agent.save_model(f"training/{self.directory}/models/agent_episode{current_episode}.pth")

    def save_episode(self):
        if self.current_episode < 1000:
            self.save_weights(self.agent_a, "a",100)
            if self.agent_b:
                self.save_weights(self.agent_b, "b",100)
        else:
            self.save_weights(self.agent_a, "a",1000)
            if self.agent_b:
                self.save_weights(self.agent_b, "b",1000)

        self.log_data_a.append([self.current_episode, self.agent_a.reward, self.agent_a.epsilon])
        if self.agent_b:
            self.log_data_b.append([self.current_episode, self.agent_b.reward, self.agent_b.epsilon])

    def learn(self):
        self.agent_a.replay()
        if self.agent_b:
            self.agent_b.replay()

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

    def save_logs(self):
        self.agent_a.save_model(f"training/{self.directory}/final_agent_episode{self.total_episodes}.pth")
        data_frame_a = pandas.DataFrame(self.log_data_a, columns=["Episode", "Reward", "Epsilon"])

        if self.agent_b:
            self.agent_b.save_model(f"training/{self.directory}/final_agent_episode{self.total_episodes}.pth")
            data_frame_b = pandas.DataFrame(self.log_data_b, columns=["Episode", "Reward", "Epsilon"])

            data_frame_a.to_csv(f"training/{self.directory}/training_log_a.csv", index=False)
            data_frame_b.to_csv(f"training/{self.directory}/training_log_b.csv", index=False)
        else:
            data_frame_a.to_csv(f"training/{self.directory}/training_log.csv", index=False)

        print("Training beendet und Log gespeichert.")

    def run_episode(self):
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

                break

            action = self.run_action(env, player)
            state = env.board.copy()

            moves += 1
            player *= -1

        self.learn()
        self.save_episode()

    def run(self):
        while self.current_episode < self.total_episodes:
            self.run_episode()
            self.current_episode += 1

        self.save_logs()
