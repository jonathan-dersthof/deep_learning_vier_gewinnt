import pandas
import copy
import os

from VierGewinnt import VierGewinnt
from Session import Session
from Agent import Agent
from logic import *

class Trainer:
    def __init__(self,
                 hidden_size,
                 state_size = 42,
                 action_size = 7,

                 gamma = 0.95,
                 epsilon = 1.0,
                 epsilon_min = 0.01,
                 epsilon_decay = 0.99995,
                 learning_rate = 0.001,
                 batch_size = 64,

                 winner_reward = 1.0,
                 draw_reward = 0.1,
                 lose_reward = -1.0,
                 survive_reward = 0.01,

                 directory = next_directory("training", "trainer"),
                 ):
        self.base_agent = Agent(state_size,
                                action_size,
                                hidden_size,

                                gamma,
                                epsilon,
                                epsilon_min,
                                epsilon_decay,
                                learning_rate,
                                batch_size,

                                winner_reward,
                                draw_reward,
                                lose_reward,
                                survive_reward, )
        self.active_agent = None

        self.directory = directory
        self.log_data = []

    def setup_trainer(self):
        if not os.path.exists(f"training/{self.directory}"):
            os.mkdir(f"training/{self.directory}")

            with open(f"training/{self.directory}/DQN_attributes", "x") as f:
                f.write("LinearDQN attributes\n"
                        f"hidden size = {self.base_agent.hidden_size}\n")

    def setup_training(self, training_type):
        new_directory = next_directory(f"training/{self.directory}", f"{training_type}_model")

        os.mkdir(f"training/{self.directory}/{new_directory}")
        os.mkdir(f"training/{self.directory}/{new_directory}/checkpoints")

        with open(f"training/{self.directory}/{new_directory}/agent_attributes", "x") as file:
            file.write("Agent attributes\n"
                    f"winner reward = {self.base_agent.winner_reward}\n"
                    f"draw reward = {self.base_agent.draw_reward}\n"
                    f"lose_reward =  {self.base_agent.lose_reward}\n"
                    f"survive_reward = {self.base_agent.survive_reward}"
                    )

        return new_directory

    def base_model(self,
                   episodes,

                   winner_reward = 1.0,
                   draw_reward = 0.1,
                   lose_reward = -1.0,
                   survive_reward = 0.01,
                   ):
        active_agent = copy.deepcopy(self.base_agent)

        self.setup_trainer()
        active_agent.set(winner_reward, draw_reward, lose_reward, survive_reward)

        new_directory = f"{self.directory}/{self.setup_training("base")}"

        training_session = Session(new_directory, episodes, active_agent)

        training_session.run()

        self.active_agent = active_agent

    def self_play(self,
                  episodes,

                  agent = None,
                  agent_directory = "",
                  winner_reward=1.0,
                  draw_reward=0.1,
                  lose_reward=-1.0,
                  survive_reward=0.01,

                  frozen = True
                  ):

        if not agent:
            agent = copy.deepcopy(self.base_agent)
            agent.load_model(agent_directory)

        agent_a = copy.deepcopy(agent)
        agent_a.epsilon = 0.25
        agent_b = copy.deepcopy(agent)
        agent_b.epsilon = 0.25

        self.setup_trainer()
        agent_a.set(winner_reward, draw_reward, lose_reward, survive_reward)
        agent_b.set(winner_reward, draw_reward, lose_reward, survive_reward)

        agent_a.epsilon_decay = 0.9995
        if frozen:
            agent_b.epsilon_min = 0
            agent_b.epsilon = 0

        new_directory = f"{self.directory}/{self.setup_training("self_play")}"

        training_session = Session(new_directory, episodes, agent_a, agent_b)

        training_session.run()

        self.active_agent = agent_a

    def full_training(self,
                      episodes,
                      cycles,

                      winner_reward=1.0,
                      draw_reward=0.1,
                      lose_reward=-1.0,
                      survive_reward=0.01,
                      ):
        self.base_model(episodes, winner_reward, draw_reward, lose_reward, survive_reward)

        for cycle in range(cycles):
            self.self_play(int(episodes/cycles), self.active_agent, "", winner_reward, draw_reward, lose_reward, survive_reward)

if __name__ == "__main__":
    new_trainer = Trainer(hidden_size = 256,
                          state_size = 42,
                          action_size = 7,

                          batch_size = 64,
                          epsilon_decay = 0.99995,

                          survive_reward = 0.02,
    )

    print(new_trainer.directory)

    new_trainer.full_training(100000, 4)
