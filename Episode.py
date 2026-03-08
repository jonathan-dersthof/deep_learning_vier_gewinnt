from Agent import Agent
from VierGewinnt import VierGewinnt
from numpy import random

import numpy

class Episode:
    """ Eine Episode beschreibt ein komplettes Spiel im Lernprozess. Eine Episode muss immer mindestens einen, kann aber auch zwei Agents enthalten. """
    def __init__(self,
                 env : VierGewinnt,
                 agent_a : Agent,
                 agent_b : Agent = None):
        self.env : VierGewinnt = env

        self.agent_a : Agent = agent_a
        self.agent_b : Agent = agent_b

        self.game_states_str : list[str] = []

    """ Grund für Berichtigung der letzten Belohnung: Siehe Agent.correct_last_reward """
    def final_rewards(self):
        """ Wendet die finalen Belohnungen auf alle Agenten an """
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

    """ Unterscheidet basierend auf den Attributen wer den Zug gerade ausführen muss """
    def run_action(self):
        """ Führt einen Spielzug aus und verteilt die Überlebensbelohnungen """
        state : numpy.ndarray = self.env.get_state()

        # Spieler 1 ist im Training immer Agent_A
        if self.env.current_player == 1:
            action = self.agent_a.act(self.env)
        elif self.agent_b:
            action = self.agent_b.act(self.env)
        else:
            action = self.env.random_move()

        self.env.step(action)
        next_state : numpy.ndarray = self.env.get_state()

        if self.env.current_player == -1:
            next_state *= -1
            state *= -1

        if self.env.current_player == 1:
            reward : float = self.agent_a.survive_reward
            self.agent_a.reward += reward

            self.agent_a.remember(state, action, reward, next_state, self.env.done)
        elif self.agent_b:
            reward : float = self.agent_b.survive_reward
            self.agent_b.reward += reward

            self.agent_b.remember(state, action, reward, next_state, self.env.done)

    def run(self):
        """ Führt eine ganze Episode aus. """
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
