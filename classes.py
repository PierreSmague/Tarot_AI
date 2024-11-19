from imports import *


class SkyjoEnv(gym.Env):
    def __init__(self, best_model):

        # Observation space
        self._obs_space_in_preferred_format = True

        spaces = {
            'observations': gym.spaces.Box(low=np.array([-2] * 26), high=np.array([12] * 26), dtype=np.int8),
            'revealed': gym.spaces.Box(low=np.array([0] * 26), high=np.array([1] * 26), dtype=np.int8)
        }

        self.observation_space = gym.spaces.Dict(spaces)

        # Action space
        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Discrete(26)
        # Initial state
        self.action_mask = np.array([1] * 2 + [0] * 24, dtype=np.int8)
        self.begin = "agent0"
        self.adversaire = "heuristic"
        self.players = ["agent0", "heuristic"]
        self.deck = [-2] * 5 + [-1] * 10 + [0] * 15
        for i in range(1, 13):
            self.deck = self.deck + [i] * 10

        self._state = {}
        # Set ending condition
        self.preums = "heuristic"
        self.best_model = best_model

        super().__init__()

    def step(self, action):

        if len(self.deck) <= 2:
            self.deck = [-2] * 5 + [-1] * 10 + [0] * 15
            for i in range(1, 13):
                self.deck = self.deck + [i] * 10

        if self._state["observations"][25] == 0:
            if action == 0:
                self._state["observations"][0] = random.choice(self.deck)
                self.deck.remove(self._state["observations"][0])
                self.action_mask = np.array([0] * 2 + [1] * 12 + [0] * 12, dtype=np.int8)
                self.action_mask[14:26] = [1 * (self._state["revealed"][i] == 0) for i in range(1, 13)]
            else:
                self.action_mask = np.array([0] * 2 + [1] * 12 + [0] * 12, dtype=np.int8)
            done = False

        else:
            if action <= 13:
                inter_card = self._state["observations"][action - 1]
                if self._state["revealed"][action - 1] == 0:
                    inter_card = random.choice(self.deck)
                    self.deck.remove(inter_card)
                self._state["observations"][action - 1] = self._state["observations"][0]
                self._state["revealed"][action - 1] = 1
                self._state["observations"][0] = inter_card
            else:
                if self._state["revealed"][action - 13] == 0:
                    self._state["observations"][action - 13] = random.choice(self.deck)
                    self._state["revealed"][action - 13] = 1
                    self.deck.remove(self._state["observations"][action - 13])

            if len([x for x in self._state["revealed"][13:25] if x == 1]) == 12:
                self.preums = "heuristic"
                done = True
            else:
                if self.adversaire == "model":
                    if self.best_model == "random":
                        self._state = skyjo_model(self._state, "random", self.deck, reverse=True)
                    else:
                        self._state = skyjo_model(self._state, self.best_model, self.deck, reverse=True)
                else:
                    self._state = heuristic_skyjo(self._state, self.deck)
                if len([x for x in self._state["revealed"][1:13] if x == 1]) == 12:
                    self.preums = "agent0"
                    done = True
                else:
                    self.action_mask = np.array([1] * 2 + [0] * 24, dtype=np.int8)
                    done = False

            for i in range(1, 5):
                if self._state["observations"][i] == self._state["observations"][i + 4] and self._state["observations"][i] == self._state["observations"][i + 8] and self._state["revealed"][i] == 1:
                    self._state["observations"][i], self._state["observations"][i + 4], self._state["observations"][i + 8] = 0, 0, 0

        self._state["observations"][25] = 1 - self._state["observations"][25]

        total_player_1 = - sum([self._state["observations"][i] for i in range(1, 13) if self._state["revealed"][i] == 1]) - sum(
            [random.choice(self.deck) for val in self._state["revealed"][1:13] if val == 0])
        total_player_2 = - sum([self._state["observations"][i] for i in range(13, 25) if self._state["revealed"][i] == 1]) - sum(
            [random.choice(self.deck) for val in self._state["revealed"][13:25] if val == 0])

        if done:
            if self.preums == "agent0":
                if total_player_1 <= total_player_2 and total_player_1 <= 0:
                    reward = 2 * total_player_1 - total_player_2
                else:
                    reward = total_player_1 - total_player_2
            else:
                if total_player_2 <= total_player_1 and total_player_2 <= 0:
                    reward = total_player_1 - 2 * total_player_2
                else:
                    reward = total_player_1 - total_player_2
        else:
            reward = 0.001 * (total_player_1 - total_player_2)

        truncated = done

        # Return step information
        return (
            self._state,
            reward,
            done,
            truncated,
            {},
        )

    def reset(self, *, seed=None, options=None):
        # Set start state
        self._state["observations"] = np.array([0] * 26, dtype=np.int8)
        self._state["revealed"] = np.array([0] * 26, dtype=np.int8)
        self.deck = [-2] * 5 + [-1] * 10 + [0] * 15
        for i in range(1, 13):
            self.deck = self.deck + [i] * 10
        self._state["observations"][0] = random.choice(self.deck)
        self.deck.remove(self._state["observations"][0])
        choix_adv = random.uniform(0, 1)
        if choix_adv < 0.25:
            self.adversaire = "heuristic"
        else:
            self.adversaire = "model"
        self.players = ["agent0"] + [self.adversaire]

        self.begin = random.choice(self.players)

        # Set ending condition
        self.preums = "heuristic"
        if self.begin != "agent0":
            if self.adversaire == "model":
                if self.best_model == "random":
                    self._state = skyjo_model(self._state, "random", self.deck, reverse=True)
                else:
                    self._state = skyjo_model(self._state, self.best_model, self.deck, reverse=True)
            else:
                self._state = heuristic_skyjo(self._state, self.deck)
        self.action_mask = np.array([1] * 2 + [0] * 24, dtype=np.int8)
        return self._state, {}

    def valid_action_mask(self):
        return self.action_mask


def skyjo_model(state, model, deck, reverse=False):
    if len(deck) <= 2:
        deck = [-2] * 5 + [-1] * 10 + [0] * 15
        for i in range(1, 13):
            deck = deck + [i] * 10

    change = 0
    if reverse:
        values = np.concatenate((state["observations"][0], state["observations"][13:25], state["observations"][1:13], state["observations"][25]), axis=None)
        revealed = np.concatenate((state["revealed"][0], state["revealed"][13:25], state["revealed"][1:13], state["revealed"][25]), axis=None)
    else:
        values = state["observations"]
        revealed = state["revealed"]

    if values[25] == 0:
        change = 1
    else:
        values[25] = 0
    action_mask = np.array([1] * 2 + [0] * 24, dtype=np.int8)
    if model == "random":
        action = random.choice([x for x in range(26) if action_mask[x] == 1])
    else:
        action = model.predict({"observations": values, "revealed": revealed}, action_masks=action_mask, deterministic=False)[0]

    if action == 0:
        values[0] = random.choice(deck)
        deck.remove(values[0])
        action_mask = np.array([0] * 2 + [1] * 12 + [0] * 12, dtype=np.int8)
        action_mask[14:26] = [1 * (revealed[i] == 0) for i in range(1, 13)]
    else:
        action_mask = np.array([0] * 2 + [1] * 12 + [0] * 12, dtype=np.int8)
    values[25] = 1

    if model == "random":
        action = random.choice([x for x in range(26) if action_mask[x] == 1])
    else:
        action = model.predict({"observations": values, "revealed": revealed}, action_masks=action_mask, deterministic=False)[0]

    if action <= 13:
        inter_card = values[action - 1]
        if revealed[action - 1] == 0:
            inter_card = random.choice(deck)
            deck.remove(inter_card)
        values[action - 1] = values[0]
        revealed[action - 1] = 1
        values[0] = inter_card
    else:
        if revealed[action - 13] == 0:
            values[action - 13] = random.choice(deck)
            revealed[action - 13] = 1
            deck.remove(values[action - 13])

    if change == 1:
        values[25] = 0

    if reverse:
        modified_values = np.concatenate((values[0], values[13:25], values[1:13], values[25]), axis=None)
        revealed = np.concatenate((revealed[0], revealed[13:25], revealed[1:13], revealed[25]), axis=None)
    else:
        modified_values = values

    return {"observations": modified_values, "revealed": revealed}


def heuristic_skyjo(state, deck):
    values = state["observations"]
    revealed = state["revealed"]

    my_revealed_cards = [values[i] for i in range(13, 25) if revealed[i]]

    max_revealed_cards, colonne = -99, 0

    if my_revealed_cards:
        max_revealed_cards = max(my_revealed_cards)
        index_max = [i for i in range(13, 25) if values[i] == max_revealed_cards][0]
    if values[0] <= 2:
        pass
    else:
        values[0] = random.choice(deck)
        deck.remove(values[0])

    if values[0] >= 2:
        possible_i = [i for i in range(1, 5) if values[i + 12] == values[i + 16] and values[i + 12] == values[0]]
        if possible_i:
            values[possible_i[0] + 12], values[possible_i[0] + 16], values[possible_i[0] + 20] = 0, 0, 0
            colonne = 1

        possible_i = [i for i in range(1, 5) if values[i + 12] == values[i + 20] and values[i + 12] == values[0]]
        if possible_i and not colonne:
            values[possible_i[0] + 12], values[possible_i[0] + 16], values[possible_i[0] + 20] = 0, 0, 0
            colonne = 1

        possible_i = [i for i in range(1, 5) if values[i + 20] == values[i + 16] and values[i + 16] == values[0]]
        if possible_i and not colonne:
            values[possible_i[0] + 12], values[possible_i[0] + 16], values[possible_i[0] + 20] = 0, 0, 0
            colonne = 1

    if not colonne:
        if max_revealed_cards != -99 and max_revealed_cards - values[0] >= 5:
            values[index_max] = values[0]
            values[0] = max_revealed_cards
        elif values[0] <= 4:
            to_reveal = random.choice([i for i in range(13, 25) if revealed[i] == 0])
            values[to_reveal] = values[0]
            revealed[to_reveal] = 1
            values[0] = random.choice(deck)
            deck.remove(values[0])
        else:
            to_reveal = random.choice([i for i in range(13, 25) if revealed[i] == 0])
            values[to_reveal] = random.choice(deck)
            revealed[to_reveal] = 1
            deck.remove(values[to_reveal])
    return {"observations": values, "revealed": revealed}


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()



def model_fight(model1, model2):
    state = {}
    state["observations"] = np.array([0] * 26, dtype=np.int8)
    state["revealed"] = np.array([0] * 26, dtype=np.int8)
    deck = [-2] * 5 + [-1] * 10 + [0] * 15
    for i in range(1, 13):
        deck = deck + [i] * 10
    state["observations"][0] = random.choice(deck)
    deck.remove(state["observations"][0])

    done, preums = False, 0
    begin = random.choice([0, 1])

    while not done:
        if begin == 0:

            state = skyjo_model(state, model1, deck)
            if len([i for i in state["revealed"][13:25] if i == 1]) == 12:
                done = True
                preums = 1
            if not done:
                if model2 == "heuristic":
                    state = heuristic_skyjo(state, deck)
                else:
                    state = skyjo_model(state, model2, deck, reverse=True)
                if len([i for i in state["revealed"][1:13] if i == 1]) == 12:
                    done = True
                    preums = 0
        else:
            if model2 == "heuristic":
                state = heuristic_skyjo(state, deck)
            else:
                state = skyjo_model(state, model2, deck, reverse=True)
            if len([i for i in state["revealed"][1:13] if i == 1]) == 12:
                done = True
                preums = 0

            if not done:
                state = skyjo_model(state, model1, deck)
                if len([i for i in state["revealed"][13:25] if i == 1]) == 12:
                    done = True
                    preums = 1

    total_player_1 = - sum([state["observations"][i] for i in range(1, 13) if state["revealed"][i] == 1]) - sum(
        [random.choice(deck) for val in state["revealed"][1:13] if val == 0])
    total_player_2 = - sum([state["observations"][i] for i in range(13, 25) if state["revealed"][i] == 1]) - sum(
        [random.choice(deck) for val in state["revealed"][13:25] if val == 0])

    if preums == 0:
        if total_player_1 <= total_player_2 and total_player_1 <= 0:
            reward = 2 * total_player_1 - total_player_2
        else:
            reward = total_player_1 - total_player_2
    else:
        if total_player_2 <= total_player_1 and total_player_2 <= 0:
            reward = total_player_1 - 2 * total_player_2
        else:
            reward = total_player_1 - total_player_2
    return reward