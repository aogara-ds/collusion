import random
import re


class Player():
    def __init__(self, name, catch_rate, agent, communicate_with=None, cool_down_period=2):
        """
        Initializes a player with the given name and identity. 
        """
        self.name = name
        self.story = ""
        self.actions = []
        self.votes = []
        self.witness = False
        self.witness_during_vote = []
        self.awaiting_response = False
        self.fishes = 0
        self.catch_rate = catch_rate
        self.isBanished = False
        self.coolDownPeriod = cool_down_period
        self.current_coolDown = 0
        self.communicate_with = None
        self.partner = communicate_with

        # Set agent and potentially model
        if "gpt" in agent:
            self.agent = "gpt"
            self.model = agent[4:]
        else:
            self.agent = agent
        assert self.agent in ["cli", "random", "gpt", "api"], \
            f"Player of type {agent} is not implemented."

        # Tracks evaluation metrics
        self.eval = {
            "name": self.name,
            "num_turns": 0,
            "banished": False,
            "fishes": self.fishes
        }

    def load_gpt(self, gpt):
        """
        Saves a reference to GPT provided by the Game class.
        """
        self.gpt = gpt

    def get_action(self, action_prompt):
        """
        Returns an integer representing a valid action based on the
        num_valid_actions argument passed into the function. 

        Part of me would prefer to read this from the player's story, 
        but maybe that's unnecessarily complicated. 
        """
        # Mark state as awaiting_response
        self.awaiting_response = True

        # Parse action prompt for valid actions
        action_int_list = [
            int(n) for n in re.findall("[0-9]",
                                       action_prompt.split("Possible Actions:")[-1])
        ]
        valid_action = False

        # Get and validate action
        while valid_action == False:
            # Get action
            if self.agent == "random":
                action_int = self.get_random_action(action_int_list)
            elif self.agent == "cli":
                action_int = self.get_cli_action(
                    action_int_list, action_prompt)
            elif self.agent == "gpt":
                action_int = self.get_gpt_action(action_prompt)

            # Validate action
            try:
                assert type(action_int) == int, \
                    "Selected action is not an integer"
                assert action_int in action_int_list, \
                    "Selected action is not in action_int_list"
                valid_action = True
            except:
                print("Invalid action. Please try again.")

        action_text = self.decode_action(action_prompt, action_int)

        # Store generated action in Player object
        self.actions.append(action_text)
        self.eval['num_turns'] += 1
        self.awaiting_response = False

        if ("Kill" in self.actions[-1]):
            self.eval['fishes'] += self.catch_rate

        return action_text

    def get_cli_action(self, action_list, action_prompt):
        # print(self.story)
        print("CLI")
        print(action_prompt)
        print(f"Please input one of the following valid inputs: {action_list}")
        return int(input())

    def get_random_action(self, action_list):
        return int(random.choice(action_list))

    def extract_list_items(self, string):
        pattern = r'(\d+)\.\s+(.*)'
        list_items = {}
        for match in re.finditer(pattern, string):
            num = int(match.group(1))
            content = match.group(2).strip()
            list_items[num] = content
        return list_items

    def get_gpt_action(self, action_prompt, argmax=False):
        action_dict = self.extract_list_items(action_prompt)
        # print(self.story + action_prompt)
        option_probs = self.gpt.get_probs(
            self.story + action_prompt, action_dict, self.model)

        if argmax:
            selected_action = max(option_probs, key=option_probs.get)
        else:
            # Sample an action from the distribution
            rand_val = random.random()
            total = 0
            for action, prob in option_probs.items():
                total += prob
                if rand_val <= total:
                    selected_action = action
                    break
            else:  # This executes if the for loop doesn't break, i.e., if no action was selected.
                selected_action = random.choice(list(option_probs.keys()))

        # Return the most likely token among the valid voting options
        return int(selected_action)

    def store_api_action(self, action_prompt, action_int):
        action_text = self.decode_action(action_prompt, action_int)
        self.actions.append(action_text)
        self.eval['num_turns'] += 1
        self.awaiting_response = False

    def decode_action(self, action_prompt, action_int):
        """
        Given an action prompt and the integer number of an action,
        returns the text description of that action.
        """
        start_description_idx = action_prompt.find(str(action_int) + ". ") + 2
        end_description_idx = action_prompt[start_description_idx:].find(
            '\n') + start_description_idx
        action_text = action_prompt[start_description_idx:end_description_idx].strip(
        )

        return action_text

    def get_statement(self, discussion_log):
        if self.agent == "random":
            statement = self.get_idk_statement()
        elif self.agent == "cli":
            statement = self.get_cli_statement(discussion_log)
        elif self.agent == "gpt":
            statement = self.get_gpt_statement(discussion_log)
        return statement + '"\n'

    def get_idk_statement(self):
        return "I don't know who the killer is."

    def get_cli_statement(self, discussion_log):
        print(self.story)
        print(discussion_log)
        return input()

    def get_gpt_statement(self, action_prompt):
        response = self.gpt.generate(
            prompt=self.story + action_prompt,
            max_tokens=50,
            model=self.model,
            # To limit GPT to providing one player's dialogue
            stop_tokens=['"']
        )
        return response
