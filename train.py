import pandas

from VierGewinnt import VierGewinnt
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
        self.active_agent = Agent(state_size,
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
                                  survive_reward,)
        self.directory = directory

    def setup_trainer(self):
        if not os.path.exists(f"training/{self.directory}"):
            os.mkdir(f"training/{self.directory}")

            with open(f"training/{self.directory}/DQN_attributes", "x") as f:
                f.write("DQN attributes\n"
                        f"hidden size = {self.active_agent.hidden_size}\n")

    def setup_training(self, training_type):
        new_directory = next_directory(f"training/{self.directory}", f"{training_type}_model")

        os.mkdir(f"training/{self.directory}/{new_directory}")
        os.mkdir(f"training/{self.directory}/{new_directory}/models")

        with open(f"training/{self.directory}/{new_directory}/agent_attributes", "x") as file:
            file.write("Agent attributes\n"
                    f"winner reward = {self.active_agent.winner_reward}\n"
                    f"draw reward = {self.active_agent.draw_reward}\n"
                    f"lose_reward =  {self.active_agent.lose_reward}\n"
                    f"survive_reward = {self.active_agent.survive_reward}"
                    )

        return new_directory

    def save_episode(self,
                     episode,
                     episodes,
                     episode_total_reward,
                     new_directory,
                     step):
            if episode % step == 0:
                print(
                    f"Episode: {episode}/{episodes}, Score: {episode_total_reward}, Epsilon: {self.active_agent.epsilon:.2f}")
                self.active_agent.save_model(
                    f"training/{self.directory}/{new_directory}/models/agent_episode{episode}.pth")

    @staticmethod
    def run_action(env, agent, player):
        state = env.board.copy()

        if player == -1:
            state *= -1

        action = agent.act(state, env)
        move = env.make_move(action, player)

        if env.check_win(move, player):
            reward = agent.winner_reward
            finished = True
        else:
            reward = agent.survive_reward
            finished = False

        next_state = env.board.copy()

        agent.remember(state, action, reward, next_state, finished)
        agent.reward += reward

        return move

    def base_model(self,
                   episodes,

                   winner_reward = 1.0,
                   draw_reward = 0.1,
                   lose_reward = -1.0,
                   survive_reward = 0.01,
                   ):
        self.setup_trainer()
        self.active_agent.set(winner_reward, draw_reward, lose_reward, survive_reward)

        new_directory = self.setup_training("base")

        log_data = []

        for episode in range(episodes):
            env = VierGewinnt()
            state = env.board.copy()

            self.active_agent.reward = 0
            moves = 0

            finished = False
            player = 1
            action = 0

            while not finished:
                if not 0 in env.get_state():
                    self.active_agent.remember(state, action, draw_reward, state, True)
                    break

                if player == 1:
                    move = self.run_action(env, self.active_agent, player)
                    action = move[0]

                    if env.check_win(move, player):
                        finished = True

                    state = env.board.copy()
                else:
                    move = env.random_move()
                    next_state = env.board.copy()

                    if env.check_win(move, player):
                        self.active_agent.remember(state, action, lose_reward, next_state, True)
                        finished = True

                    state = next_state

                moves += 1
                player *= -1

            self.active_agent.replay()
            log_data.append([episode, self.active_agent.reward, self.active_agent.epsilon])

            if episode < 1000:
                self.save_episode(episode, episodes, self.active_agent.reward, new_directory, 100)
            else:
                self.save_episode(episode, episodes, self.active_agent.reward, new_directory, 1000)

        self.active_agent.save_model(f"training/{self.directory}/{new_directory}/final_agent_episode{episode}.pth")
        data_frame = pandas.DataFrame(log_data, columns=["Episode", "Reward", "Epsilon"])
        data_frame.to_csv(f"training/{self.directory}/{new_directory}/training_log.csv", index=False)
        print("Training beendet und Log gespeichert.")

"""    def self_play(self,
                  episodes,

                  winner_reward=1.0,
                  draw_reward=0.1,
                  lose_reward=-1.0,
                  survive_reward=0.01,
                  ):
        new_directory = next_directory(self.directory, "self_play")
        os.mkdir(f"training/{self.directory}/{new_directory}")
        os.mkdir(f"training/{self.directory}/{new_directory}/models")

        with open(f"training/{self.directory}/{new_directory}/agent_attributes", "x") as f:
            f.write("Agent attributes\n"
                    f"winner reward = {winner_reward}\n"
                    f"draw reward = {draw_reward}\n"
                    f"lose_reward =  {lose_reward}\n"
                    f"survive_reward = {survive_reward}"
                    )

        log_data = []

        for episode in range(episodes):
            env = VierGewinnt()
            state = env.board.copy()

            episode_total_reward = 0
            moves = 0

            finished = False
            player = 1

            while not finished:
                if not 0 in env.get_state():
                    self.active_agent.remember(state, action, draw_reward, state, True)

                    if moves < 1:
                        print("unentschieden ohne züge")
                    break

                if player == 1:
                    action = self.active_agent.act(state, env)
                    move = env.make_move(action, player)

                    if move:
                        if env.check_win(move, player):
                            reward = winner_reward
                            finished = True
                        else:
                            reward = survive_reward

                        next_state = env.board.copy()

                    self.active_agent.remember(state, action, reward, next_state, finished)
                    state = next_state
                    episode_total_reward += reward
                else:
                    valid_moves = [c for c in range(7) if env.board[0, c] == 0]
                    random_col = valid_moves[numpy.random.randint(0, len(valid_moves))]

                    move = env.make_move(random_col, player)
                    next_state = env.board.copy()

                    if move:
                        if env.check_win(move, player):
                            self.active_agent.remember(state, action, lose_reward, env.board.copy(), True)
                            finished = True

                    state = next_state

                moves += 1
                player *= -1

            self.active_agent.replay()

            log_data.append([episode, episode_total_reward, self.active_agent.epsilon])

            if episode < 1000:
                if episode % 100 == 0:
                    print(
                        f"Episode: {episode}/{episodes}, Score: {episode_total_reward}, Epsilon: {self.active_agent.epsilon:.2f}")
                    self.active_agent.save_model(
                        f"training/{self.directory}/{new_directory}/models/agent_episode{episode}.pth")
            else:
                if episode % 1000 == 0:
                    print(
                        f"Episode: {episode}/{episodes}, Score: {episode_total_reward}, Epsilon: {self.active_agent.epsilon:.2f}")
                    self.active_agent.save_model(
                        f"training/{self.directory}/{new_directory}/models/agent_episode{episode}.pth")

        self.active_agent.save_model(f"training/{self.directory}/{new_directory}/final_agent_episode{episode}.pth")
        data_frame = pandas.DataFrame(log_data, columns=["Episode", "Reward", "Epsilon"])
        data_frame.to_csv(f"training/{self.directory}/{new_directory}/training_log.csv", index=False)
        print("Training beendet und Log gespeichert.")"""

if __name__ == "__main__":
    new_trainer = Trainer(hidden_size = 256,
                          state_size = 42,
                          action_size = 7,

                          batch_size = 128,
                          epsilon_decay = 0.999,

                          survive_reward = 0.0001,
                          directory = "trainer_17"
    )

    print(new_trainer.directory)

    new_trainer.base_model(episodes = 100000)
    new_trainer.base_model(episodes = 100000)
    #new_trainer.base_model(episodes = 100000)
