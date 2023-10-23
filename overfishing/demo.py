from agent import Player
from environment import Game


# n_games = 10
# for i in range(1, n_games+1):
#     # Define the game
#     game = Game()

#     # Load the players into the game
#     game.load_players([
#         Player("Bob", catch_rate=1, agent="gpt-3.5",
#                communicate_with=None, cool_down_period=2),
#         Player("Adam", catch_rate=5, agent="gpt-3.5",
#                communicate_with="Charlie", cool_down_period=2),
#         Player("Charlie", catch_rate=3,  agent="gpt-3.5",
#                communicate_with="Adam", cool_down_period=2),
#     ])
#     # Play the game
#     print("called")
#     game.play(n_game=i)

game = Game()
# Load the players into the game
game.load_players([
    Player("Bob", catch_rate=3, agent="gpt-3.5",
           communicate_with=None, cool_down_period=2),
    Player("Adam", catch_rate=5, agent="gpt-3.5",
           communicate_with="Charlie", cool_down_period=2),
    Player("Charlie", catch_rate=3,  agent="gpt-3.5",
           communicate_with="Adam", cool_down_period=2),
])
game.play(1)
