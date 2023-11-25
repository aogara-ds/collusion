from agent import Player
from gpt import GPT

import pandas as pd
import numpy as np
import os

import json


class Game():
    # Logs JSON
    prompts_json = {}
    throw_stats_temp = {}
    round_logs = {}

    # Self Record_csv_logs
    rounds_data = []



    def __init__(self, discussion=False, n_rounds=5):
        print("Initialized game.")
        self.discussion = discussion
        self.prompts = self.load_prompts()
        self.n_rounds = n_rounds
        self.player_stories = {}
        self.throw_stats_temp = {}
        self.rounds_dict = {}

    def load_players(self, players, bots=0):
        """
        Loads specific players with defined names and identities
        """
        # Initialize list of players
        self.players = players

        # Initializing the rounds dict
        for player in self.players:
            self.rounds_dict[player.name] = 0
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
        print("Banished PLayers list", players)
        for player in players:
            player.current_coolDown += 1
            player.isBanished = player.current_coolDown % player.coolDownPeriod == 0

    def communicate_partners(self, round_n):
        communicate_text_json = {}
        players = list(filter(lambda x: x.partner != None, self.players))
        for player in players:
            if (player.partner not in communicate_text_json):
                text_str = self.player_stories[player.name] + self.prompts_json["initiate_communication"].format(
                    PLAYER=player.partner
                )
            else:
                text_str = self.player_stories[player.name] + self.prompts_json["initiated_comm_reply"].format(
                    PLAYER1=player.partner,
                    PLAYER1_MESSAGE=communicate_text_json[player.partner]
                )
            communicate_text_json[player.name] = player.get_statement(
                text_str).split(":")[-1]

        # Update Communication_logs
        for player in players:
            player.communication_logs[round_n] = {}
            player.communication_logs[round_n]["from"] = communicate_text_json[player.name]
            player.communication_logs[round_n]["to"] = communicate_text_json[player.partner]

    def format_communication_logs(self, player):
        if (player.partner == None):
            return ""
        log_str = "These are the communication logs you had with your partner " + \
            player.partner + "\n"
        for round_n, convo in player.communication_logs.items():
            log_str += self.prompts_json["log_template"].format(
                PLAYER=player.name,
                TEXT=convo["from"]
            ) + self.prompts_json["log_template"].format(
                PLAYER=player.partner,
                TEXT=convo["to"]
            )
        # print(log_str)
        return log_str

    def print_stats(self, round_n):
        print("---------------------")
        print("ROUND :", round_n)
        for player in self.players:
            print("{PLAYER}:{SCORE}".format(
                PLAYER=player.name, SCORE=player.fishes))
        print("---------------------")

    def play(self, n_game):
        print("Game : ", str(n_game))

        for i in range(self.n_rounds):
            self.print_stats(i)

            self.round_logs[i] = []

            self.update_banished_players()
            cur_active_players = self.get_active_players()
            if (self.discussion == True):
                self.communicate_partners(i)

            # Initial Decisions
            curr_decision_dict = {}
            throw_votes = {}
            for player in cur_active_players:
                throw_votes[player.name] = []
            # throw_votes = self.throw_stats_temp.copy()

            # Accumulation of round moves
            for player in cur_active_players:

                # Action request for player
                print(player.name + " decision making")
                story = ""

                story = self.generate_rules_set(player)
                story += self.generate_rounds_prompt(i-1)
                story += self.format_communication_logs(player)
                story += self.generate_game_round_prompt(player)
                story += self.prompts_json["action_prompt"].format(
                    POSSIBLE_ACTIONS=self.load_actions(player))
                final_prompt = self.player_stories[player.name] + self.generate_game_round_prompt(
                    player) + \
                    self.prompts_json["action_prompt"].format(
                        POSSIBLE_ACTIONS=self.load_actions(player))
                curr_decision_dict[player.name] = {"player_obj": player, "action": player.get_action(
                    final_prompt)}

                if ("THROW" in curr_decision_dict[player.name]["action"]):
                    throw_votes[curr_decision_dict[player.name]["action"].split("THROW:")[
                        1].strip()].append(player.name)

                self.player_stories[player.name] = story
                player.player_story+=story

            print(throw_votes)

            # Round Moves Execution
            skip_moves = []
            throw_votes_keys_desc = sorted(throw_votes, key=lambda key: len(
                throw_votes[key]), reverse=True)
            throw_votes = {key: throw_votes[key]
                           for key in throw_votes_keys_desc}
            allthrew = list(map(lambda x: len(x), list(
                throw_votes.values()))) == np.ones(len(self.players)).tolist()

            if (allthrew):
                self.round_logs[i].append(self.prompts_json["everyone_fought"])
                continue

            for player, enemies in throw_votes.items():
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
                        player_obj.current_coolDown += 1
                    elif (len(enemies) == 1):
                        self.round_logs[i].append(self.prompts_json["tried_failed"].format(
                            PLAYER_1=enemies[0],
                            THROWN_PLAYER=player
                        ))
                        skip_moves += [enemies[0]]
                    else:
                        self.round_logs[i].append(self.prompts_json["fishing"].format(
                            PLAYER_NAME=player
                        ))
                        player_obj.fishes += player_obj.catch_rate

            self.record_round(n_round=i)

        self.save_csv(n_game)

    def record_round(self, n_round):
        for player in self.players:
            self.rounds_data.append({
                "round": n_round,
                "Player Name": player.name,
                "# Round in Water": player.current_coolDown,
                "# Fishes Caught": player.fishes,
                "story": str(player.player_story)
            })

    def save_csv(self,n_game):
        file_name = "final_results_temp.csv"
        df = pd.DataFrame()
        if (os.path.exists(file_name)):
            df = pd.read_csv(file_name)

        df1 = pd.DataFrame(self.rounds_data)
        df1['communication'] = {True:"YES",False:"NO"}[self.discussion]
        df1['Game'] = str(n_game)
        df1_new = pd.concat([df,df1])
        df1_new.to_csv("final_results_temp.csv",index=False)
        print("Temporary File Saved")

    def format_actions(self, actions):
        formatted_actions = ""
        for i, a in enumerate(actions):
            formatted_actions += f"\n{i+1}. {a}"
        return formatted_actions

    def load_actions(self, player):
        actions = ["FISH"]
        # If the player is the killer, allow them to kill opponents in their location
        players = self.get_active_players()

        players = list(filter(lambda x: x.isBanished ==
                              False and x.name != player.name, players))

        actions.extend(
            ["THROW:" + o.name for o in players])

        # Allow the player to escape through the unlocked door if applicable
        return self.format_actions(actions)

    def generate_rounds_prompt(self, round_inx):
        prompt = ""
        for inx in range(round_inx+1):
            prompt = prompt+self.prompts_json["round_text"].format(
                ROUND_NO=str(inx),
                TEXT=",".join(self.round_logs[inx])
            )+"\n"
        return prompt

    def generate_rules_set(self, player):
        names = list(filter(lambda x: x != player.name, [
            player.name for player in self.players]))
        rules_set_prompt = self.prompts_json["rules_set"].format(
            NAME=player.name,
            NAME_2=names[0],
            NAME_3=names[1],
            X=len(self.players),
            Y=player.catch_rate

        )
        non_partner = {True: names[1], False: names[0]}[
            names[0] == player.partner]
        rules_set_prompt += "\n" + \
            {True: self.prompts_json["partner_prompt"].format(
                PARTNER_NAME=player.partner, non_partner_NAME=non_partner), False: self.prompts_json["no_partner_prompt"]}[player.partner != None]

        return rules_set_prompt+"\n"

    def generate_game_round_prompt(self, player):
        prompt_template = self.prompts_json["player_status"]
        prompt = [prompt_template.format(
            PLAYER_NAME=player.name, FISHES_CAUGHT=player.fishes) + "\n" for player in self.players]
        return "".join(prompt)+"\n"

    def load_prompts(self):
        """
        Returns a dictionary of the main prompts used by the game.
        """
        # Opening JSON file
        f = open('./prompts.json', "rb")
        self.prompts_json = json.load(f)

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
