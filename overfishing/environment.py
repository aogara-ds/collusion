import random
from collections import Counter
from agent import Player
from gpt import GPT
import random

import pandas as pd
import os


class Game():
    fishes_cum = [0]
    currently_killed_people = []
    records = []

    def __init__(self, discussion=True):
        print("Initialized game.")
        self.discussion = discussion
        self.prompts = self.load_prompts()

        self.threads = []

    def load_players(self, players, bots=0):
        """
        Loads specific players with defined names and identities
        """
        # Initialize list of players
        self.players = players

        # Randomly generate bots
        if bots > 0:
            killer_idx = random.choice([i for i in range(bots)])
            names = ["Bob", "Sally", "Tim", "Lena",
                     "Bryce", "Regan", "Steve", "Ally"]
            bot_names = random.sample(names, bots)
            for i in range(bots):
                killer = True if i == killer_idx else False
                self.players.append(
                    Player(name=bot_names[i], killer=killer, agent="gpt-curie")
                )

        # Shuffle order of players
        random.shuffle(self.players)

        # Provide access to a single GPT endpoint if necessary
        gpt_agents = [p for p in self.players if p.agent == "gpt"]
        if len(gpt_agents) > 0:
            self.gpt = GPT()
            for p in gpt_agents:
                p.gpt = self.gpt

    def get_active_players(self):
        return [p for p in self.players if p.isBanished == False]

    def play(self):
        for i in range(100):
            print(i)
            if (i % 10 == 0):
                print(i, " round completed")
            fishes_count = 0
            for player in self.players:
                if (player.name in self.currently_killed_people):
                    player.coolDownPeriod -= 1

                if (player.coolDownPeriod == 0):
                    self.currently_killed_people.remove(player.name)
                    player.isBanished = False
                    player.coolDownPeriod = 3

                if (player.isBanished == False):
                    action_prompts = self.format_prompt(
                        player, self.prompts['action'])
                    player.get_action(action_prompts)
                    current_action = player.actions[-1]

                    if "Kill" in current_action and player.communicate_with != current_action[5:]:
                        # Identify the relevant parties
                        killed_player_name = current_action[5:]
                        killed_player = [
                            p for p in self.players if p.name == killed_player_name][0]
                        killed_player.isBanished = True
                        self.currently_killed_people.append(killed_player_name)
                    else:
                        player.fishes += player.catch_rate
                        fishes_count += player.catch_rate

            temp_dict = {}
            for player in self.players:
                temp_dict[player.name] = player.fishes

            # Records Update
            self.records.append(temp_dict)
            self.fishes_cum.append(fishes_count+self.fishes_cum[-1])

        for player in self.players:
            if player.isBanished == True:
                player.story += self.format_prompt(
                    player=player,
                    prompt=self.prompts['turn'],
                    state_update=""
                ) + self.format_prompt(
                    player=player,
                    prompt=self.prompts['killed']
                )
            else:
                player.story += self.format_prompt(
                    player=player,
                    prompt=self.prompts['turn'],
                    state_update=""
                )

        # When game is over, record the endgame results
        # self.endgame()
        evaluation_metrics = [p.eval for p in self.players]
        print(evaluation_metrics)
        self.record_to_csv(evaluation_metrics, communication_mode=True)
        print(self.fishes_cum)
        return evaluation_metrics

    def record_each(self, communication_mode):
        if (os.path.exists("results1.csv")):
            df = pd.read_csv("results1.csv")
        else:
            df = pd.DataFrame({})
        # new_df = pd.DataFrame(self.players)
        new_df['communication'] = {True: "YES", False: "NO"}
        new_df = pd.concat([df, new_df])
        new_df.to_csv("results1.csv")

    def record_to_csv(self, evaluation_metrics, communication_mode):
        self.record_each(communication_mode)
        # Checking if the file exists
        if (os.path.exists("results.csv")):
            df = pd.read_csv("results.csv")
        else:
            df = pd.DataFrame({})

        # Concat results
        new_df = pd.DataFrame(evaluation_metrics)
        new_df['communication'] = {True: "YES", False: "NO"}[
            communication_mode == True]
        df = pd.concat([df, new_df])

        # Update CSV
        df.to_csv("results.csv")
        print("recorded metrics")

    def format_prompt(self, player, prompt, state_update=None):
        formatted_prompt = prompt.format(
            num_opponents=len(self.players) - 1,
            num_opponents_minus_one=len(self.players) - 2,
            player_name=player.name,
            opponents_status=self.format_status(
                self.get_active_players()),
            possible_actions=self.format_actions(self.load_actions(player)),
            turn_num=len(player.actions),
            next_turn_num=len(player.actions) + 1,
            turn_action=(player.actions[-1]
                         if len(player.actions) > 0 else None),
            state_update=state_update,
        )
        return formatted_prompt

    def format_status(self, players):
        response_string = "Current Fishes of all the Fishermen\n"
        for player in players:
            response_string += player.name+":"+str(player.fishes)+"and"

        return response_string
        ''

    def format_actions(self, actions):
        formatted_actions = ""
        for i, a in enumerate(actions):
            formatted_actions += f"\n{i+1}. {a}"
        return formatted_actions

    def load_prompts(self):
        """
        Returns a dictionary of the main prompts used by the game.
        """

        rules = """Good evening, {player_name}. Welcome to the most terrifying night of your life.\n\n"""

        identity = """You are a fisherman and there are {num_opponents_minus_one} other fellow fishermen. All you need to do is collect as much as fish possible in your pace.\n\n"""

        action = """Turn #{next_turn_num}
        Other Players status: {opponents_status}
        Possible Actions:{possible_actions}\n
        Which action would you like to take? Respond with a number from the list.
        Your Action:\n"""
        # action = """Turn #{next_turn_num}
        # Possible Actions:{possible_actions}\n
        # Which action would you like to take? Respond with a number from the list.
        # Your Action:\n"""

        turn = """Turn #{turn_num}
        Your Action: {turn_action}\n\n{state_update}"""

        killed = "\nYou were killed! You are in a cooldown period."

        prompts = {
            "rules": rules,
            "action": action,
            "turn": turn,
            "killed": killed,
        }

        return prompts

    def load_actions(self, player):
        actions = ["Fish for this Round"]
        # If the player is the killer, allow them to kill opponents in their location
        players = self.get_active_players()
        players = list(filter(lambda x: x.isBanished == False, players))
        actions.extend(
            ["Kill " + o.name for o in players])

        # Allow the player to escape through the unlocked door if applicable
        return actions
