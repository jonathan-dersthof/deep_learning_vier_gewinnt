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
        self.directory = directory
        self.log_data = []

    def setup_trainer(self):
        if not os.path.exists(f"training/{self.directory}"):
            os.mkdir(f"training/{self.directory}")

            with open(f"training/{self.directory}/DQN_attributes", "x") as f:
                f.write("DQN attributes\n"
                        f"hidden size = {self.base_agent.hidden_size}\n")

    def setup_training(self, training_type):
        new_directory = next_directory(f"training/{self.directory}", f"{training_type}_model")

        os.mkdir(f"training/{self.directory}/{new_directory}")
        os.mkdir(f"training/{self.directory}/{new_directory}/models")

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

        training_session = Session(new_directory, active_agent)

        training_session.run(episodes)

        active_agent.save_model(f"training/{new_directory}/final_agent_episode{episodes}.pth")
        data_frame = pandas.DataFrame(training_session.log_data, columns=["Episode", "Reward", "Epsilon"])
        data_frame.to_csv(f"training/{new_directory}/training_log.csv", index=False)
        print("Training beendet und Log gespeichert.")


if __name__ == "__main__":
    new_trainer = Trainer(hidden_size = 256,
                          state_size = 42,
                          action_size = 7,

                          batch_size = 128,
                          epsilon_decay = 0.999,

                          survive_reward = 0.0001,
    )

    print(new_trainer.directory)

    new_trainer.base_model(episodes = 100000)
    new_trainer.base_model(episodes = 100000)
    #new_trainer.base_model(episodes = 100000)
