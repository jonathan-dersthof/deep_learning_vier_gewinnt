from VierGewinnt import VierGewinnt
from Agent import Agent

from logic import select_agent

class Game:
    def __init__(self,
                 player1 : Agent | str = None,
                 player2 : Agent | str = None):
        self.env = VierGewinnt()

        self.players = {1: player1, -1: player2}
        self.current_player = None

        self.score = {1 : 0,
                      -1 : 0
                      }

    def select_player(self, player : str):
        while True:
            user_input = input(f"Spieler 1:\n1. KI\n2. Mensch\n> ")

            if user_input == "1" or user_input.lower() == "ki":
                selected_player = select_agent()
                selected_player.epsilon = 0.1
                break
            elif user_input == "2" or user_input.lower() == "mensch":
                selected_player = input("Namen eingeben\n> ")
                break
            else:
                print("Option 1 oder 2 wählen.")

        match player:
            case "Spieler 1":
                self.players[1] = selected_player

            case "Spieler 2":
                self.players[-1] = selected_player

    def select_players(self):
        self.select_player("Spieler 1")
        self.select_player("Spieler 2")

    def show_score(self):
        for i in [1, -1]:
            if isinstance(self.players[i], Agent):
                print(f"KI {self.env.players[i]} Score: {self.score[i]}")
            else:
                print(f"{self.env.players[i]} '{self.players[i]}' Score: {self.score[i]}")
        print("")

    def end_screen(self):
        self.env.show_board()

        match self.env.outcome:
            case "draw":
                print("Unentschieden")

            case "win":
                if isinstance(self.current_player, Agent):
                    print(f"KI {self.env.players[self.env.current_player]} hat gewonnen")
                else:
                    print(f"{self.env.players[self.env.current_player]} '{self.current_player}' hat gewonnen\n")

    def get_human_move(self):
         while True:
            try:
                action = int(input(f"{self.env.players[self.env.current_player]} '{self.current_player}', wähle Spalte (1-7): ")) - 1
                if action in self.env.get_valid_moves():
                    return action
                print("Ungültiger Zug! Spalte voll oder außerhalb des Bereichs.")

            except ValueError:
                print("Bitte eine Zahl eingeben.")

    def run(self):
        self.env.show_board()

        while not self.env.done:
            self.current_player = self.players[self.env.current_player]

            if isinstance(self.current_player, Agent):
                action = self.current_player.act(self.env)
                print(f"\nKI {self.env.players[self.env.current_player]} Zug: {action + 1}")
            else:
                action = self.get_human_move()
                print(f"\n{self.env.players[self.env.current_player]} '{self.current_player}' Zug: {action + 1}")

            self.env.step(action)
            self.env.show_board()

            if not self.env.done:
                self.env.current_player *= -1

    def play(self) -> dict:
        if not self.players[1] and not self.players[-1]:
            self.select_players()

        self.env.reset()
        self.env.current_player = 1

        self.run()

        if self.env.outcome == "win":
            self.score[self.env.current_player] += 1

        self.end_screen()
        self.show_score()

        while True:
            user_input = input(f"Wie geht es weiter? \n1. Neue Runde? \n2. Beenden?\n")

            if user_input == "1" or user_input.lower() == "neue runde":
                return self.play()
            elif user_input == "2" or user_input.lower() == "beenden":
                return self.score
            else:
                print("Option 1 oder 2 wählen.")
