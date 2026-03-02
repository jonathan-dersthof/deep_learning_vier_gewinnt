import pandas

class Logger:
    def __init__(self, directory : str):
        self.log_data_a = []
        self.log_data_b = []

        self.directory = directory

        self.last_win_rate_a = 0.0
        self.last_win_rate_b = 0.0

    def save_checkpoint(self, session, step):
        current_episode = session.current_episode_number
        total_episodes = session.total_episodes

        if current_episode % step == 0:
            print(f"Episode: {current_episode}/{total_episodes}, Score: {session.agent_a.reward}, Epsilon: {session.agent_a.epsilon:.2f}")
            self.save_replay(session)

            if session.agent_b_pool and len(session.agent_b_pool) == 1 and session.agent_b_pool[0].epsilon != 0:
                session.agent_a.save_model(f"training/{self.directory}/checkpoints/agent_a_episode{current_episode}.pth")
                session.agent_b_pool[0].save_model(f"training/{self.directory}/checkpoints/agent_b_episode{current_episode}.pth")
            else:
                session.agent_a.save_model(f"training/{self.directory}/checkpoints/agent_episode{current_episode}.pth")

    def save_replay(self, session):
        current_episode = session.current_episode_number

        with open(f"training/{self.directory}/checkpoints/game_episode{current_episode}.txt", "x") as file:
            if session.agent_b_pool:
                file.write(f"Anzahl Agenten im Agentenpool: {len(session.agent_b_pool)}\n Aktueller Agent: {session.current_agent_b_number} \n")
            for state in session.current_game:
                file.write(f"{state}\n")

    def save_episode(self, session):
        current_episode = session.current_episode_number
        total_episodes = session.total_episodes

        if current_episode < 1000:
            self.save_checkpoint(session, 100)
        else:
            self.save_checkpoint(session, 1000)

        if current_episode + 1 == total_episodes:
            session.agent_a.save_model(f"training/{self.directory}/final_agent_a_episode{total_episodes}.pth")
            if session.agent_b_pool and len(session.agent_b_pool) == 1 and session.agent_b_pool[0].epsilon != 0:
                session.agent_b_pool[0].save_model(f"training/{self.directory}/final_agent_b_episode{total_episodes}.pth")

    def log_episode(self, session, new_win_rate_a=None, new_win_rate_b=None):
        if new_win_rate_a:
            self.last_win_rate_a = new_win_rate_a

        self.log_data_a.append([
            session.current_episode_number,
            session.agent_a.reward,
            session.agent_a.epsilon,
            session.agent_a.current_loss,
            self.last_win_rate_a
        ])

        if session.agent_b_pool and len(session.agent_b_pool) == 1:
            if new_win_rate_b:
                self.last_win_rate_b = new_win_rate_b
            agent_b = session.agent_b_pool[0]
            self.log_data_b.append([
                session.current_episode_number,
                agent_b.reward,
                agent_b.epsilon,
                agent_b.current_loss,
                self.last_win_rate_b
            ])

    def save_logs_to_csv(self):
        columns = ["Episode", "Reward", "Epsilon", "Loss", "WinRate"]
        data_frame_a = pandas.DataFrame(self.log_data_a, columns=columns)

        if len(self.log_data_b):
            data_frame_a.to_csv(f"training/{self.directory}/training_log_a.csv", index=False)

            data_frame_b = pandas.DataFrame(self.log_data_b, columns=columns)
            data_frame_b.to_csv(f"training/{self.directory}/training_log_b.csv", index=False)
        else:
            data_frame_a.to_csv(f"training/{self.directory}/training_log.csv", index=False)

        print("Trainingssession beendet und Log gespeichert.")
