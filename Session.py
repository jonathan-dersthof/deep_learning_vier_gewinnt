import pandas
import numpy

from VierGewinnt import VierGewinnt
from Agent import Agent
from logic import *
from Episode import Episode
from Logger import Logger


class Session:
    def __init__(self,
                 directory : str,
                 total_episodes : int,
                 agent_a : Agent,
                 agent_b : list[Agent] = None
                 ):
        self.agent_a = agent_a
        self.agent_b_pool = agent_b
        self.current_agent_b_number  = None

        self.env = VierGewinnt()
        self.directory = directory

        self.log_data_a = []
        self.log_data_b = []

        self.total_episodes = total_episodes
        self.current_episode_number = 0
        self.current_episode = None

        self.current_game = []

        self.logger = Logger(directory)

    def learn(self):
        self.agent_a.replay()

        if self.agent_b_pool and len(self.agent_b_pool) == 1 and self.agent_b_pool[0].epsilon != 0:
            self.agent_b_pool[0].replay()

    def evaluate_agent(self, agent, num_test_games=100):
        wins = 0
        draws = 0
        losses = 0

        saved_epsilon = agent.epsilon
        agent.epsilon = 0.0

        for i in range(num_test_games):
            self.env.reset()

            while not self.env.done:
                if self.env.current_player == 1:
                    action = agent.act(self.env)
                else:
                    action = self.env.random_move()

                self.env.step(action)

                if not self.env.done:
                    self.env.current_player *= -1

            if self.env.outcome == "win":
                if self.env.current_player == 1:
                    wins += 1
                else:
                    losses += 1
            elif self.env.outcome == "draw":
                draws += 1

        agent.epsilon = saved_epsilon

        win_rate = (wins + 0.5 * draws) / num_test_games
        self.env.reset()

        return win_rate

    def run_episode(self):
        if not self.agent_b_pool:
            episode = Episode(self.env, self.agent_a)
        else:
            if len(self.agent_b_pool) == 1:
                self.current_agent_b_number = 0
            else:
                self.current_agent_b_number = numpy.random.randint(len(self.agent_b_pool))

            agent_b = self.agent_b_pool[self.current_agent_b_number]
            episode = Episode(self.env, self.agent_a, agent_b)

        episode.run()
        self.current_game = episode.game_states_str

        self.learn()

        new_win_rate_a = None
        new_win_rate_b = None

        if self.current_episode_number % 100 == 0:
            new_win_rate_a = self.evaluate_agent(self.agent_a)

            if self.agent_b_pool and len(self.agent_b_pool) == 1 and self.agent_b_pool[0].epsilon != 0:
                agent_b = self.agent_b_pool[0]
                new_win_rate_b = self.evaluate_agent(agent_b)

        self.logger.log_episode(self, new_win_rate_a, new_win_rate_b)

        self.logger.save_episode(self)
        self.env.reset()

    def run(self):
        while self.current_episode_number < self.total_episodes:
            self.run_episode()
            self.current_episode_number += 1

        self.logger.save_logs_to_csv()
