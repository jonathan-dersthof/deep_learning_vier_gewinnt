import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as numpy

from VierGewinnt import VierGewinnt
from collections import deque
from LinearDQN import LinearDQN


class Agent:
    """ Die Agent-Klasse verwaltet das DQN Modell und beinhaltet alle Methoden für das Lernen und Handeln des Agenten """
    def __init__(self,
                 hidden_size : int = 256,

                 gamma : float = 0.95,
                 epsilon : float = 1.0,
                 epsilon_min : float = 0.01,
                 epsilon_decay : float = 0.99995,
                 learning_rate : float = 0.001,
                 batch_size : int = 64,

                 winner_reward : float = 1.0,
                 draw_reward : float = 0.1,
                 lose_reward : float = -1.0,
                 survive_reward : float = 0.01
                 ):
        self.hidden_size : int = hidden_size

        self.memory : deque = deque(maxlen=100000)

        self.gamma : float= gamma
        self.epsilon : float = epsilon
        self.epsilon_min : float = epsilon_min
        self.epsilon_decay : float = epsilon_decay
        self.learning_rate : float = learning_rate
        self.batch_size : int = batch_size

        self.model : nn.Module = LinearDQN(hidden_size)
        self.target_model : nn.Module = LinearDQN(hidden_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Schrittweite in welcher target_model aktualisiert wird
        self.update_step : int= 1000
        self.steps : int = 0

        self.optimizer : optim.Adam = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion : nn.SmoothL1Loss = nn.SmoothL1Loss()

        self.win_reward : float = winner_reward
        self.draw_reward : float = draw_reward
        self.lose_reward : float = lose_reward
        self.survive_reward : float = survive_reward

        # gesammelter Reward innerhalb einer Episode
        self.reward : float = 0.0
        self.current_loss : float = 0.0

    def set_hyperparameters(self,
                            gamma: float,
                            epsilon: float,
                            epsilon_min: float,
                            epsilon_decay: float,
                            learning_rate: float,
                            batch_size: int,

                            win_reward: float,
                            draw_reward: float,
                            lose_reward: float,
                            survive_reward: float
                            ):
        """ Setzt die Hyperparameter des Agenten neu"""
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.lose_reward = lose_reward
        self.survive_reward = survive_reward

    def remember(self,
                 state : numpy.ndarray,
                 action : int,
                 reward : float,
                 next_state : numpy.ndarray,
                 done : bool):
        """ Fügt der Erinnerung des Agenten einen weiteren Datensatz hinzu. """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, env : VierGewinnt) -> int:
        """ Die Methode wählt den nächsten Zug aus einer Liste an legalen Zügen auf dem Brett aus. """
        state : numpy.ndarray = env.get_state()

        # state Normalisierung: der Agent sieht das Spielbrett immer aus der Sicht von Spieler 1
        # → dadurch muss der Agent nicht lernen zu unterscheiden welcher Spieler er ist
        if env.current_player == -1:
            state *= -1

        valid_moves = env.get_valid_moves()

        if not valid_moves:
            return 0

        # exploration
        if numpy.random.rand() <= self.epsilon:
            return random.choice(valid_moves)

        state_t : torch.Tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            act_values = self.model(state_t)

        q_values = act_values[0].cpu().numpy().copy()

        # illegale Züge erhalten einen Q-Wert in der negativen Unendlichkeit, da in Randerscheinungen im Training trotz einem vorher gewählten sehr niedrigen Wert illegale Züge gewählt wurden
        for c in range(7):
            if env.board[0, c] != 0:
                q_values[c] = -numpy.inf

        # wählt alle möglichen Aktionen mit dem gleichen Maximal vorausgesagtem Q-Wert als potenzielle Aktionen aus
        max_q = numpy.max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]

        return random.choice(best_actions)

    """ Um den Agent klarer vom Environment zu trennen, wurde die Reward Logik aus der VierGewinnt  Klasse nach Episode verschoben.
    Da in diesem Rahmen jeder überlebte Zug als survive_reward gewertet wird, muss nach Spielende die letzte Erinnerung überschrieben werden, um win_reward etc zu verteilen"""
    def correct_last_reward(self, reward : float):
        """ Korrigiert letzten Eintrag der Erinnerung. """
        if self.memory:
            last_memory = list(self.memory[-1])
            last_memory[2] = reward
            last_memory[4] = True
            self.memory[-1] = tuple(last_memory)
            self.reward += reward

    def replay(self):
        """ In der Methode findet die Auswertung der letzten Episode statt. """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(numpy.array([x[0] for x in minibatch]))
        actions = torch.LongTensor([x[1] for x in minibatch])
        rewards = torch.FloatTensor([x[2] for x in minibatch])
        next_states = torch.FloatTensor(numpy.array([x[3] for x in minibatch]))
        dones = torch.FloatTensor([x[4] for x in minibatch])

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            max_next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1

        if self.steps % self.update_step == 0:
           self.target_model.load_state_dict(self.model.state_dict())

        self.current_loss = loss.item()

    """ nur model weights werden gespeichert/geladen, nicht ganzer Agent """
    def save_model(self, path : str):
        """ Speichert die Model-Weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Modell gespeichert unter {path}")

    def load_model(self, path : str):
        """ Lädt und evaluiert die Model-Weights"""
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())
        self.model.eval()
        print(f"Modell aus '{path}' geladen.")
