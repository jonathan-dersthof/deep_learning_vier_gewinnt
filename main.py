from VierGewinnt import VierGewinnt
import numpy
from Agent import Agent

def main():
    players = {
        1 : "Spieler 1",
        -1 : "Spieler 2",
    }
    vg = VierGewinnt()
    ki = Agent(state_size=42, action_size=7, hidden_size=128)

    model1 = "models/agent_1_ep99900.pth"

    model2 = "training/trainer/base_model/final_agent_episode100000.pth"
    model3 = "training/trainer/self_play_model_3/models/agent_a_episode24000.pth"

    ki.load_model(model3)
    ki.epsilon = 0.0
    player = 1

    while True:
        if player == 1:
            action = ki.act(vg.get_state(), vg)
            vg.show_board()
            move = vg.make_move(action, player)
        else:
            vg.show_board()
            col = int(input(f"Spieler {players[player]} ist dran:"))-1
            move = vg.make_move(col, player)

        if move:
            if not 0 in vg.get_state():
                print("Unentschieden")
                break
            elif vg.check_win(move, player):
                print(f"Gewonnen {players[player]}!")
                vg.show_board()
                break

            player *= -1


if __name__ == "__main__":
    main()
