from agent import Player
from environment import Game

# Define the game
game = Game()

# Load the players into the game
game.load_players([
    Player("Bob", catch_rate=3, agent="gpt-3.5"),
    Player("Adam", catch_rate=5, agent="gpt-3.5", communicate_with="Charlie"),
    Player("Charlie", catch_rate=1,  agent="gpt-3.5", communicate_with="Adam"),
])
# Play the game
game.play()
