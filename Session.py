import pandas
import numpy

from VierGewinnt import VierGewinnt
from Agent import Agent
from logic import *
from Episode import Episode


class Session:
    def __init__(self,
                 directory,
                 episodes,
                 agent_a,
                 agent_b = None
                 ):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.current_agent_b = None

        self.directory = directory

        self.log_data_a = []
        self.log_data_b = []

        self.current_episode = 0
        self.total_episodes = episodes

        self.current_game = []

    def save_checkpoint(self, step):
        if self.current_episode % step == 0:
            print(f"Episode: {self.current_episode}/{self.total_episodes}, Score: {self.agent_a.reward}, Epsilon: {self.agent_a.epsilon:.2f}")
            self.save_game()

            if self.agent_b and len(self.agent_b) == 1 and self.agent_b[0].epsilon != 0:
                self.agent_a.save_model(f"training/{self.directory}/checkpoints/agent_a_episode{self.current_episode}.pth")
                self.agent_b[0].save_model(f"training/{self.directory}/checkpoints/agent_b_episode{self.current_episode}.pth")
            else:
                self.agent_a.save_model(f"training/{self.directory}/checkpoints/agent_episode{self.current_episode}.pth")

    def save_game(self):
        with open(f"training/{self.directory}/checkpoints/game_episode{self.current_episode}.txt", "x") as file:
            if self.agent_b:
                file.write(f"Number of opponents: {len(self.agent_b)}\n Current Agent: {self.current_agent_b} \n")
            for state in self.current_game:
                file.write(f"{state}\n")

    def save_episode(self):
        if self.current_episode < 1000:
            self.save_checkpoint(100)
        else:
            self.save_checkpoint(1000)

        self.log_data_a.append([self.current_episode, self.agent_a.reward, self.agent_a.epsilon])
        if self.agent_b and len(self.agent_b) == 1:
            self.log_data_b.append([self.current_episode, self.agent_b[0].reward, self.agent_b[0].epsilon])

    def save_logs(self):
        data_frame_a = pandas.DataFrame(self.log_data_a, columns=["Episode", "Reward", "Epsilon"])

        if self.agent_b and len(self.agent_b) == 1:
            data_frame_a.to_csv(f"training/{self.directory}/training_log_a.csv", index=False)
            self.agent_a.save_model(f"training/{self.directory}/final_agent_a_episode{self.total_episodes}.pth")

            data_frame_b = pandas.DataFrame(self.log_data_b, columns=["Episode", "Reward", "Epsilon"])
            data_frame_b.to_csv(f"training/{self.directory}/training_log_b.csv", index=False)
            self.agent_b[0].save_model(f"training/{self.directory}/final_agent_b_episode{self.total_episodes}.pth")
        else:
            data_frame_a.to_csv(f"training/{self.directory}/training_log.csv", index=False)
            self.agent_a.save_model(f"training/{self.directory}/final_agent_episode{self.total_episodes}.pth")

        print("Training beendet und Log gespeichert.")

    def learn(self):
        self.agent_a.replay()

        if self.agent_b and len(self.agent_b) == 1 and self.agent_b[0].epsilon != 0:
            self.agent_b[0].replay()

    def run_episode(self):
        if not self.agent_b:
            episode = Episode(self.agent_a)
        elif len(self.agent_b) == 1:
            episode = Episode(self.agent_a, self.agent_b[0])
        else:
            self.current_agent_b = numpy.random.randint(len(self.agent_b))
            episode = Episode(self.agent_a, self.agent_b[self.current_agent_b])

        episode.run()
        self.current_game = episode.game_states_str

        self.learn()
        self.save_episode()

    def run(self):
        while self.current_episode < self.total_episodes:
            self.run_episode()
            self.current_episode += 1

        self.save_logs()
