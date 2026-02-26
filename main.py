import numpy

from VierGewinnt import VierGewinnt
from Agent import Agent
from Game import Game

"""def main():
    players = {
        1 : "Spieler 1",
        -1 : "Spieler 2",
    }
    vg = VierGewinnt()
    ki = Agent(state_size=42, action_size=7, hidden_size=256)

    model1 = "models/agent_1_ep99900.pth"

    model2 = "training/trainer_3/base_model/final_agent_episode100000.pth"
    model3 = "training/trainer_18/self_play_model_3/final_agent_episode25000.pth"

    ki.load_model(model3)
    ki.epsilon = 0.1
    vg.current_player = 1

    while True:
        if vg.current_player == 1:
            action = ki.act(vg)
            vg.show_board()
            move = vg.make_move(action)
        else:
            vg.show_board()
            col = int(input(f"{players[vg.current_player]} ist dran:"))-1
            move = vg.make_move(col)

        if move:
            if not 0 in vg.get_state():
                print("Unentschieden")
                break
            elif vg.check_win(move):
                print(f"Gewonnen {players[vg.current_player]}!")
                vg.show_board()
                break

            vg.current_player *= -1"""

def main():
    model1 = "models/agent_1_ep99900.pth"

    model2 = "training/trainer_3/base_model/final_agent_episode100000.pth"
    model3 = "training/trainer_18/self_play_model_3/final_agent_episode25000.pth"

    ki = Agent(state_size=42, action_size=7, hidden_size=256)

    ki.load_model(model3)
    ki.epsilon = 0.1

    game = Game("Johny", ki)
    game.play()

if __name__ == "__main__":
    main()
