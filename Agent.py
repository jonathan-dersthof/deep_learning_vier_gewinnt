import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy

from collections import deque
from DeepQNetwork import DQN

class Agent:
    def __init__(self,
                 state_size = 42,
                 action_size = 7,
                 hidden_size = 256,

                 gamma = 0.95,
                 epsilon = 1.0,
                 epsilon_min = 0.01,
                 epsilon_decay = 0.99995,
                 learning_rate = 0.001,
                 batch_size = 64,

                 winner_reward = 1.0,
                 draw_reward = 0.1,
                 lose_reward = -1.0,
                 survive_reward = 0.01
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.memory = deque(maxlen=100000)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.model = DQN(hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.winner_reward = winner_reward
        self.draw_reward = draw_reward
        self.lose_reward = lose_reward
        self.survive_reward = survive_reward

        self.reward = 0

    def set(self,
            winner_reward,
            draw_reward,
            lose_reward,
            survive_reward
    ):
        self.winner_reward = winner_reward
        self.draw_reward = draw_reward
        self.lose_reward = lose_reward
        self.survive_reward = survive_reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env):
        valid_moves = [c for c in range(7) if env.board[0, c] == 0]

        if not valid_moves:
            return 0

        if numpy.random.rand() <= self.epsilon:
            return random.choice(valid_moves)

        state_t = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            act_values = self.model(state_t)

        q_values = act_values[0].cpu().numpy().copy()

        for c in range(7):
            if env.board[0, c] != 0:
                q_values[c] = -numpy.inf

        max_q = numpy.max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]

        return random.choice(best_actions)

    def replay(self):
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
            max_next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Modell gespeichert unter {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Modell aus '{path}' geladen.")
