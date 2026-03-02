import numpy
import os

from VierGewinnt import VierGewinnt
from Agent import Agent
from Game import Game

from logic import select_agent

def main():
    game = Game()
    game.play()

if __name__ == "__main__":
    main()
