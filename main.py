from VierGewinnt import VierGewinnt
import numpy
from Agent import Agent

def main():
    players = {
        1 : "Spieler 1",
        -1 : "Spieler 2",
    }
    vg = VierGewinnt()
    ki = Agent(state_size=42, action_size=7, hidden_size=256)

    model1 = "models/agent_1_ep99900.pth"

    model2 = "training/trainer_16/base_model_2/final_agent_episode99999.pth"
    model3 = "training/trainer_16/base_model_3/final_agent_episode99999.pth"
    model4 = "training/trainer_16/base_model_4/final_agent_episode99999.pth"

    model5 = "training/trainer_17/base_model/final_agent_episode99999.pth"
    model6 = "training/trainer_17/base_model_1/final_agent_episode99999.pth"
    model7 = "training/trainer_17/base_model_2/final_agent_episode99999.pth"

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
            #col = numpy.random.randint(7)
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
