import copy
import os
import pandas

from Session import Session
from Agent import Agent
from logic import next_directory
from Logger import Logger

class Trainer:
    """ Verwaltet mehrere Trainings-sessions von Agenten mit gleichem DQN"""
    def __init__(self,
                 hidden_size : int,
                 directory : str  = next_directory("training", "trainer")
                 ):
        self.agents : list[Agent]= []
        self.agents.append(Agent(hidden_size = hidden_size))

        if directory.startswith("training"):
            self.directory : str = directory[len("training"):]
        else:
            self.directory = directory

    def setup_trainer(self):
        """ Erstellt einen Trainer_n Ordner inklusive txt Datei mit DQN Attributen, falls nötig. """
        if not os.path.exists(f"training/{self.directory}"):
            os.mkdir(f"training/{self.directory}")

            with open(f"training/{self.directory}/DQN_attributes", "x") as f:
                f.write("LinearDQN attributes\n"
                        f"hidden size = {self.agents[0].hidden_size}\n")

    def setup_training(self, training_type : str, agent : Agent):
        """ Erstellt Ordner für aktuelle Trainingssession inklusive txt Datei mit Hyperparametern des Agenten"""
        new_directory : str = next_directory(f"training/{self.directory}", f"{training_type}_model")

        os.mkdir(f"training/{self.directory}/{new_directory}")
        os.mkdir(f"training/{self.directory}/{new_directory}/checkpoints")

        with open(f"training/{self.directory}/{new_directory}/agent_attributes", "x") as file:
            file.write("Agent attributes\n"
                    f"winner reward = {agent.win_reward}\n"
                    f"draw reward = {agent.draw_reward}\n"
                    f"lose_reward =  {agent.lose_reward}\n"
                    f"survive_reward = {agent.survive_reward}"
                    )

        return new_directory

    """ Basis Modell trainiert gegen Zufallsgegner, damit kein anderes fertiges Modell benötigt wird und weil Min-Max 
    potentiell zu schwer wäre und Modell dauerhaft bestraft werden würde und somit kaum lernt """
    def base_training(self,
                      episodes : int,

                      gamma : float = 0.95,
                      epsilon : float = 1.0,
                      epsilon_min : float = 0.01,
                      epsilon_decay : float = 0.99995,
                      learning_rate : float = 0.001,
                      batch_size : int = 64,

                      win_reward: float = 1.0,
                      draw_reward: float = 0.1,
                      lose_reward: float = -1.0,
                      survive_reward: float = 0.01
                      ):
        """ Trainiert ein Model von Grund auf gegen einen Zufallsgegner (kein externes Modell nötig)"""
        # agents[0] existiert immer, weil ein Agent im Konstruktor initialisiert wird
        base_agent : Agent = self.agents[0]

        self.setup_trainer()
        base_agent.set_hyperparameters(gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, batch_size, win_reward, draw_reward, lose_reward, survive_reward)

        new_directory = f"{self.directory}/{self.setup_training("base", base_agent)}"

        training_session : Session = Session(new_directory, episodes, base_agent)

        training_session.run()

        return training_session.logger.log_data_a

    def league_play(self,
                    episodes : int,

                    agent : Agent | None = None,
                    agent_directory : str | None = None,

                    gamma : float = 0.95,
                    epsilon : float = 1.0,
                    epsilon_min : float = 0.01,
                    epsilon_decay : float = 0.99995,
                    learning_rate : float = 0.001,
                    batch_size : int = 64,

                    win_reward : float = 1.0,
                    draw_reward : float = 0.1,
                    lose_reward : float = -1.0,
                    survive_reward : float = 0.01
                    ):
        """ Agent trainiert gegen alle vorherigen finalen Modelle eines Trainingsprozesses """
        if not agent:
            agent : Agent = copy.deepcopy(self.agents[0])

            if agent_directory:
                agent.load_model(agent_directory)

        agent_a : Agent = agent
        agent_a.epsilon = 0.25
        agent_a.epsilon_decay = 0.99995
        agent_a.set_hyperparameters(gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, batch_size, win_reward, draw_reward, lose_reward, survive_reward)

        agent_pool : list[Agent] = []

        for i in range(len(self.agents)):
            agent_pool.append(copy.deepcopy(self.agents[i]))

            agent_pool[i].epsilon_min = 0
            agent_pool[i].epsilon = 0

        self.setup_trainer()

        new_directory : str = f"{self.directory}/{self.setup_training("league_play", agent_a)}"

        training_session : Session = Session(new_directory, episodes, agent_a, agent_pool = agent_pool)
        training_session.run()

        # Agent wird den verwalteten Agents des Trainers hinzugefügt (nötig, weil agent_a eine deepcopy ist, anders als in base_training()
        self.agents.append(agent_a)

        return training_session.logger.log_data_a

    def self_play(self,
                  episodes : int,

                  agent_a : Agent | None = None,
                  agent_a_directory : str | None = None,
                  agent_b : Agent | None = None,
                  agent_b_directory : str | None = None,

                  gamma : float = 0.95,
                  epsilon : float = 1.0,
                  epsilon_min : float = 0.01,
                  epsilon_decay : float = 0.99995,
                  learning_rate : float = 0.001,
                  batch_size : int = 64,

                  win_reward : float = 1.0,
                  draw_reward : float = 0.1,
                  lose_reward : float = -1.0,
                  survive_reward : float = 0.01
                  ):
        """ Zu trainierender wird kopiert und spielt gegen sich selbst. Die kopierte Version darf potenziell auch lernen """
        pass

    """ Für base- und league_training() entschieden, da nur base_training() gegen einen Menschen kaum eine chance hat und reines self play zu instabil werden kann """
    def full_training(self,
                      episodes : int,
                      cycles : int,

                      gamma : float = 0.95,
                      epsilon : float = 1.0,
                      epsilon_min : float = 0.01,
                      epsilon_decay : float = 0.99995,
                      learning_rate : float = 0.001,
                      batch_size : int = 64,

                      win_reward : float = 1.0,
                      draw_reward : float = 0.1,
                      lose_reward : float = -1.0,
                      survive_reward : float = 0.01
                      ):
        """ Führt ein 'vollständiges Training' bestehend aus erst base- und dann league_training() durch, das ein einigermaßen kompetentes Modell erzeugt.  """
        logger = Logger(self.directory)

        columns = ["Episode", "Reward", "Epsilon", "Loss", "WinRate"]
        data : list[pandas.DataFrame] = [pandas.DataFrame(self.base_training(episodes, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, batch_size, win_reward, draw_reward, lose_reward, survive_reward), columns=columns)]

        for cycle in range(cycles):
           data.append(pandas.DataFrame(self.league_play(int((cycle + 1) * (episodes / 10)), self.agents[-1], "", gamma, epsilon / 10, epsilon_min, epsilon_decay, learning_rate, batch_size, win_reward, draw_reward, lose_reward, survive_reward), columns=columns))

        combined_data : pandas.DataFrame = pandas.concat(data, ignore_index=True)
        logger.plot_overview(combined_data)

if __name__ == "__main__":
    new_trainer = Trainer(hidden_size = 256)

    print(new_trainer.directory)

    new_trainer.full_training(100000, 5,
                              win_reward=1.0,
                              draw_reward=0.1,
                              lose_reward=-1.0,
                              survive_reward=-0.005
                              )
