from Game import Game
from Training import Training

def main():
    while True:
        user_input = input(f"Möchtest du ein Modell trainieren oder spielen?:\n1. spielen\n2. trainieren\n> ")

        if user_input == "1" or "sp" in user_input.lower():
            game = Game()
            game.play()
            break
        elif user_input == "2" or "tra" in user_input.lower():
            training = Training()
            training.train()
            break
        else:
            print("Option 1 oder 2 wählen.")

if __name__ == "__main__":
    main()
