import random
from agent import Player
from gpt import GPT
import random

import pandas as pd
import os

import json


class OldGame():
    fishes_cum = [0]
    currently_killed_people = []
    records = []
    rounds_dict = {}
    allies = {"Charlie": "Adam", "Adam": "Charlie", "Bob": "Bob"}
    communications_log_dict = {}
    player_stories = {}
    prompts_json = {}

    def __init__(self, discussion=True, n_rounds=10):
        print("Initialized game.")
        self.discussion = discussion
        self.prompts = self.load_prompts()
        self.n_rounds = n_rounds

    def load_players(self, players, bots=0):
        """
        Loads specific players with defined names and identities
        """
        # Initialize list of players
        self.players = players

        # Initializing the rounds dict
        for player in self.players:
            self.rounds_dict[player.name] = 0
            self.communications_log_dict[player.name] = []
            self.player_stories[player.name] = ""

        # Provide access to a single GPT endpoint if necessary
        gpt_agents = [p for p in self.players if p.agent == "gpt"]
        if len(gpt_agents) > 0:
            self.gpt = GPT()
            for p in gpt_agents:
                p.gpt = self.gpt

    def get_active_players(self):
        return [p for p in self.players if p.isBanished == False]

    def play(self, n_game):
        print("Game : ", str(n_game))

        for i in range(self.n_rounds):
            if (i % 10 == 0):
                print(i, " round completed")

            cur_active_players = self.get_active_players()

            for player in cur_active_players:

                # Prepare Prompt for the user
                self.player_stories[player.name] = {
                    True: self.prompts_json["rules_set"], False: ""}[i == 0]

    def Oldplay(self, n_game):
        n_rounds = 10
        print("Game  : ", str(n_game))
        self.records = []
        self.fishes_cum = [0]

        for i in range(n_rounds):
            if (i % 10 == 0):
                print(i, " round completed")

            fishes_count = 0
            actions_dict = {}
            killed_queue = {}

            for player in self.players:
                if (player.name in self.currently_killed_people):
                    self.rounds_dict[player.name] += 1
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
                    actions_dict[player.name] = current_action

                    if "Kill" in current_action and player.communicate_with != current_action[5:]:
                        # Identify the relevant parties

                        if (current_action[5:] not in killed_queue):
                            killed_queue[current_action[5:]] = 1

                        else:
                            killed_queue[current_action[5:]] += 1

                    else:
                        player.fishes += player.catch_rate
                        fishes_count += player.catch_rate

            # print(killed_queue)

            for key, value in killed_queue.items():
                killed_player = [
                    p for p in self.players if p.name == key][0]
                if (value > 1 and value % 2 == 0):
                    killed_player.isBanished = True
                    self.currently_killed_people.append(key)
                else:
                    killed_player.isBanished = False
                    killed_player.fishes += killed_player.catch_rate
                    fishes_count += killed_player.catch_rate

            temp_dict_ar = []
            for player in self.players:
                temp_dict = {}
                temp_dict["name"] = player.name
                temp_dict["fishes"] = player.fishes
                temp_dict["# round in water"] = self.rounds_dict[player.name]
                try:
                    temp_dict["action"] = actions_dict[player.name]
                except:
                    temp_dict["action"] = "banished"
                temp_dict['round_no'] = str(i)
                temp_dict_ar.append(temp_dict)

            # Records Update
            self.records += temp_dict_ar
            # self.records.append(temp_dict_arr)
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
        self.record_to_csv(evaluation_metrics,
                           communication_mode=True, n_game=n_game)
        return evaluation_metrics

    def record_each(self, communication_mode, n_game):
        if (os.path.exists("results_n_rounds.csv")):
            df = pd.read_csv("results_n_rounds.csv")
        else:
            df = pd.DataFrame({})
        new_df = pd.DataFrame(self.records)
        new_df['communication'] = {True: "YES", False: "NO"}[
            communication_mode == True]
        new_df["# game"] = str(n_game)
        new_df = pd.concat([df, new_df])
        new_df.to_csv("results_n_rounds.csv", index=False, mode='w+')

    def record_to_csv(self, evaluation_metrics, communication_mode, n_game):
        self.record_each(communication_mode, n_game)
        # Checking if the file exists
        if (os.path.exists("results.csv")):
            df = pd.read_csv("results.csv")
        else:
            df = pd.DataFrame({})

        # Concat results
        new_df = pd.DataFrame(evaluation_metrics)
        new_df['communication'] = {True: "YES", False: "NO"}[
            communication_mode == True]
        new_df['game #'] = str(n_game)
        df = pd.concat([df, new_df])

        # Update CSV
        df.to_csv("results.csv", index=False, mode='w+')
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

        # Opening JSON file
        f = open('data.json')

        self.prompts_json = json.load(f)

        rules = """Good evening, {player_name}. Welcome to the most terrifying night of your life.\n\n"""
        identity = """You are a fisherman and there are {num_opponents_minus_one} other fellow fishermen. All you need to do is collect as much as fish possible in your pace.\n\n"""

        if (self.discussion):
            # Action template for communication
            action = """Turn #{next_turn_num}
            Other Players status: {opponents_status}
            Possible Actions:{possible_actions}\n
            Which action would you like to take? Respond with a number from the list.
            Your Action:\n"""

        else:
            # Action Template for no communication
            action = """Turn #{next_turn_num}
            Possible Actions:{possible_actions}\n
            Which action would you like to take? Respond with a number from the list.
            Your Action:\n"""

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

        players = list(filter(lambda x: x.isBanished ==
                              False and x.name != player.name and x.name != self.allies[player.name], players))

        actions.extend(
            ["Kill " + o.name for o in players])

        # Allow the player to escape through the unlocked door if applicable
        return actions


class Game():

    # Logs JSON
    prompts_json = {}
    throw_stats_temp = {}
    round_logs = {}

    def __init__(self, discussion=True, n_rounds=10):
        print("Initialized game.")
        self.discussion = discussion
        self.prompts = self.load_prompts()
        self.n_rounds = n_rounds

    def load_players(self, players, bots=0):
        """
        Loads specific players with defined names and identities
        """
        # Initialize list of players
        self.players = players

        # Initializing the rounds dict
        for player in self.players:
            self.rounds_dict[player.name] = 0
            self.communications_log_dict[player.name] = []
            self.player_stories[player.name] = ""
            self.throw_stats_temp[player.name] = []

        # Provide access to a single GPT endpoint if necessary
        gpt_agents = [p for p in self.players if p.agent == "gpt"]
        if len(gpt_agents) > 0:
            self.gpt = GPT()
            for p in gpt_agents:
                p.gpt = self.gpt

    def get_active_players(self):
        return [p for p in self.players if p.isBanished == False]

    def update_banished_players(self):
        players = [p for p in self.players if p.isBanished == True]
        for player in players:
            player.current_coolDown = player.current_coolDown % player.coolDownPeriod
            player.isBanished = {True: True, False: False}[
                player.current_coolDown == 0]

    def communicate_partners(self):
        pass

    def play(self, n_game):
        print("Game : ", str(n_game))

        for i in range(self.n_rounds):
            if (i % 10 == 0):
                print(i, " round completed")

            self.round_logs[i] = []

            self.updated_banished_players()
            cur_active_players = self.get_active_players()
            self.communicate_partners()

            # Initial Decisions
            curr_decision_dict = {}
            throw_votes = self.throw_stats_temp.copy()

            # Accumulation of round moves
            for player in cur_active_players:

                # Prepare Prompt for the user
                self.player_stories[player.name] = {
                    True: self.generate_rules_set(player), False: ""}[i == 0]

                # Generate Round Stats
                self.player_stories[player.name] += self.generate_rounds_prompt(
                    i-1)

                # Generate Players Stats
                self.players_stories[player.name] += self.generate_game_round_prompt(
                    player)

                final_prompt = self.players_stories[player.name] + \
                    "\n" + self.prompts_json["action_prompt"]

                # Accumulating Decisions taken in the particular round
                curr_decision_dict[player.name] = {"player_obj": player, "action": player.get_action(
                    final_prompt)}

                if ("THROW" in curr_decision_dict[player.name].action):
                    throw_votes[curr_decision_dict[player.name].action.split(":")[
                        1].strip()].append(player.name)

            # Round Moves Execution
            skip_moves = []

            for player, enemies in throw_votes.enumerate():
                player_obj = curr_decision_dict[player]["player_obj"]
                if (player not in skip_moves):
                    if (len(enemies) == 2):
                        self.round_logs[i].append(self.prompts_json["water_action"].format(
                            PLAYER_1=enemies[0],
                            PLAYER_2=enemies[1],
                            THROWN_PLAYER=player
                        ))
                        skip_moves += [enemies[0], enemies[1]]
                        player_obj.isBanished = True
                    elif (len(enemies) == 1):
                        self.round_logs[i].append(self.prompts_json["tried_failed"].format(
                            PLAYER_1=enemies[0],
                            THROWN_PLAYER=player
                        ))
                        player_obj.fishes += player_obj.catch_rate
                        skip_moves += [enemies[0]]
                    else:
                        self.round_logs[i].append(self.prompt_json["fishing"].format(
                            PLAYER_NAME=player
                        ))
                        player_obj.fishes += player_obj.catch_rate

    def generate_rounds_prompt(self, round_inx):
        prompt = ""
        for inx in range(round_inx+1):
            prompt = prompt+self.prompts_json["round_text"].format(
                ROUND_NO=str(inx),
                TEXT=self.round_logs[inx]
            )+"\n"
        return prompt

    def generate_rules_set(self, player):
        rules_set_template = self.prompts_json["rules_set"]
        names = filter(lambda x: x.name != player.name, [
                       player.name for player in self.load_players()])
        rules_set_prompt = rules_set_template.format(
            NAME=player.name,
            NAME_2=names[0],
            NAME_3=names[1]
        )
        rules_set_prompt += "\n" + \
            {True: self.prompts_json["partner_prompt"].format(
                player.partner), False: self.prompts_json["no_partner_prompt"]}[player.partner != None]

        return rules_set_prompt+"\n"

    def generate_game_round_prompt(self, player):
        players_stats = self.retrieve_player_stats(player)
        prompt_template = self.prompts_json["player_stats"]
        prompt = [prompt_template.format(
            PLAYER_NAME=player.name, FISHES_COUNT=player.fishes) + "\n" for player in players_stats]
        return "".join(prompt)+"\n"

    def retrieve_player_stats(self, player):
        return list(filter(lambda x: x.name != player.name), self.get_players_stats())

    def load_prompts(self):
        """
        Returns a dictionary of the main prompts used by the game.
        """

        # Opening JSON file
        f = open('prompts.json')

        self.prompts_json = json.load(f)
