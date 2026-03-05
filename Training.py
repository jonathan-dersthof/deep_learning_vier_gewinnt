from logic import select_path, next_directory, get_attributes_from_file, select_value, select_agent
from Trainer import Trainer
from Agent import Agent


class Training:
    def __init__(self):
        self.trainer = None

    def setup(self):
        while True:
            user_input = input(f"Trainer wählen oder neuen Trainer erstellen?:\n1. Trainer Wählen\n2. Neuer Trainer\n> ")

            if user_input == "1" or "wählen" in user_input.lower():
                selected_trainer = select_path("training", "trainer_")
                hidden_size = get_attributes_from_file(f"{selected_trainer}/DQN_attributes")["hidden size"]
                break
            elif user_input == "2" or "neu" in user_input.lower():
                selected_trainer = next_directory("training", "trainer")
                hidden_size = self.select_value_default("hidden size wählen", 256, "int")
                break
            else:
                print("Option 1 oder 2 wählen.")

        self.trainer = Trainer(hidden_size = hidden_size,
                               directory = selected_trainer
                               )

    @staticmethod
    def select_value_default(prompt, default = None, value_type = None):
        user_input = select_value(prompt, default = default, value_type = value_type)

        if user_input:
            return user_input
        else:
            return default

    def get_hyperparameters(self):
        default_agent = Agent()
        print("Hyperparameter festlegen: ")
        epsilon = self.select_value_default("Epsilon", default = default_agent.epsilon, value_type = "float")
        epsilon_min = self.select_value_default("Minimalwert für Epsilon (epsilon_min)", default = default_agent.epsilon_min, value_type = "float")
        epsilon_decay = self.select_value_default("Epsilon Verfallsrate (epsilon_decay)", default = default_agent.epsilon_decay, value_type = "float")
        learning_rate = self.select_value_default("Lernrate (learning_rate)", default = default_agent.learning_rate, value_type = "float")
        batch_size = self.select_value_default("Lernrate (learning_rate)", default = default_agent.batch_size, value_type = "int")

        print("Belohnungen festlegen für: ")
        win_reward = self.select_value_default("Sieg(win_reward)", default = default_agent.win_reward, value_type = "float")
        draw_reward = self.select_value_default("Unentschieden (draw_reward)", default = default_agent.draw_reward, value_type = "float")
        lose_reward = self.select_value_default("Niederlage (lose_reward)", default = default_agent.lose_reward, value_type = "float")
        survive_reward = self.select_value_default("Überleben (survive_reward)", default = default_agent.survive_reward, value_type = "float")

        return {
            "epsilon" : epsilon,
            "epsilon_min" : epsilon_min,
            "epsilon_decay" : epsilon_decay,
            "learning_rate" : learning_rate,
            "batch_size" : batch_size,
            "win_reward" : win_reward,
            "draw_reward" : draw_reward,
            "lose_reward" : lose_reward,
            "survive_reward" : survive_reward
            }

    @staticmethod
    def select_training():
        while True:
            while True:
                user_input = input(f"Training Wählen:\n1. Basistraining\n2. Self-play\n3. League-play\n4. Volles Training\n> ")

                if user_input == "1" or "basis" in user_input.lower():
                    return "base_training"
                elif user_input == "2" or "self" in user_input.lower():
                    return "self_play"
                elif user_input == "3" or "league" in user_input.lower():
                    return "league_play"
                elif user_input == "4" or "voll" in user_input.lower():
                    return "full_training"
                else:
                    print("Option 1 - 4 wählen.")

    def train(self):
        if not self.trainer:
            self.setup()

        selected_training = self.select_training()
        hyperparameters = self.get_hyperparameters()
        episodes = select_value("Episodenanzahl festlegen: ", value_type = "int")

        match selected_training:
            case "base_training":
                self.trainer.base_training(episodes, **hyperparameters)
            case "self_play":
                self.trainer.self_play(episodes, **hyperparameters)
            case "league_play":
                agent_a = select_agent()
                self.trainer.league_play(episodes, agent_a, None, **hyperparameters)
            case "full_training":
                cycles = select_value("Anzahl an Trainingswiederholungen (cycles) festlegen: ", value_type = "int")
                self.trainer.full_training(episodes, cycles, **hyperparameters)

if __name__ == "__main__":
    training = Training()
    training.train()
