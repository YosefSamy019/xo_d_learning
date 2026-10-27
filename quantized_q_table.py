import json
import os
import numpy as np
import pandas as pd

_GETTER_COUNT_KEY = '#op getter count'
_SETTER_COUNT_KEY = '#op setter count'

class QuantizedQTable:
    def __init__(self,
                 states_max_vals,
                 states_min_vals,
                 n_actions,
                 n_quantization_lvls,
                 cache_folder,
                 state_names=None,
                 action_names=None,
                 use_access_counter=False,
                 ):

        self.state_length = len(states_max_vals)
        self.states_max_vals = np.array(states_max_vals)
        self.states_min_vals = np.array(states_min_vals)
        self.n_actions = int(n_actions)

        if type(n_quantization_lvls) is int or type(n_quantization_lvls) is float:
            self.n_quantization_lvls = [int(n_quantization_lvls)] * self.state_length
        else:
            self.n_quantization_lvls = np.array(n_quantization_lvls)

        os.makedirs(cache_folder, exist_ok=True)
        self.cache_path = os.path.join(cache_folder, "qtable.json")
        self.csv_export_path = os.path.join(cache_folder, "qtable.csv")

        self._q_table = self._load_from_cache(fallback=dict())
        self._states_pins = self._create_states_pins()

        if state_names is None:
            state_names = [f"state {i}" for i in range(self.state_length)]
        self._state_names = state_names

        if action_names is None:
            action_names = [f'action {i}' for i in range(self.n_actions)]
        self._action_names = action_names

        self.use_access_counter = use_access_counter

    def _load_from_cache(self, fallback):
        if os.path.exists(self.cache_path):
            print("Cache file exists, loading from it.")
            with open(self.cache_path, "r") as f:
                return json.load(f)
        else:
            print("Cache file does not exist, creating it.")
            return fallback

    def save_to_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self._q_table, f)

    def _create_states_pins(self):
        states_pins = []

        for i in range(self.state_length):
            states_pins.append(
                np.linspace(self.states_min_vals[i], self.states_max_vals[i], self.n_quantization_lvls[i]))

        return states_pins

    def quantize_state(self, state):
        quantized_state = np.zeros((self.state_length,))
        quantized_indices = np.zeros((self.state_length,))

        for i in range(self.state_length):
            state_pin_i = self._states_pins[i]
            indices_i = np.digitize(state[i], state_pin_i) - 1
            indices_i = np.clip(indices_i, 0, self.n_quantization_lvls[i] - 1)

            quantized_state[i] = state_pin_i[indices_i]
            quantized_indices[i] = indices_i

        return quantized_state, quantized_indices

    def _get_state_id(self, quantized_indices):
        return "#".join(map(lambda x: str(int(x)), quantized_indices))

    def _on_access_listener(self, state_id, apply_get=False, apply_set=False):
        access_dict = self._q_table[state_id]

        if apply_get:
            access_dict[_GETTER_COUNT_KEY] += 1

        if apply_set:
            access_dict[_SETTER_COUNT_KEY] += 1

    def set_val(self, state, action_id, val):
        quantized_state, quantized_indices = self.quantize_state(state)
        state_id = self._get_state_id(quantized_indices)

        if state_id not in self._q_table:
            accessed_dict = self._randomize_entry(quantized_state)
            accessed_dict[self._action_names[action_id]] = float(val)
            self._q_table[state_id] = accessed_dict
        else:
            self._q_table[state_id][self._action_names[action_id]] = float(val)

        self._on_access_listener(state_id, apply_set=True)

    def get_val(self, state, action_id):
        quantized_state, quantized_indices = self.quantize_state(state)
        state_id = self._get_state_id(quantized_indices)

        if state_id not in self._q_table:
            self._q_table[state_id] = self._randomize_entry(quantized_state)

        self._on_access_listener(state_id, apply_get=True)

        return self._q_table[state_id][self._action_names[action_id]]

    def _randomize_entry(self, quantized_state):
        accessed_dict = dict()

        accessed_dict[_GETTER_COUNT_KEY] = 0
        accessed_dict[_SETTER_COUNT_KEY] = 0

        for i, v in enumerate(quantized_state):
            state_name_i = self._state_names[i]
            accessed_dict[state_name_i] = float(v)

        for i in range(self.n_actions):
            action_name_i = self._action_names[i]
            accessed_dict[action_name_i] = np.random.uniform(-0.001, 0.001)

        return accessed_dict

    def get_vals(self, state):
        quantized_state, quantized_indices = self.quantize_state(state)
        state_id = self._get_state_id(quantized_indices)

        if state_id not in self._q_table:
            self._q_table[state_id] = self._randomize_entry(quantized_state)

        accessed_dict = self._q_table[state_id]
        return_val = np.zeros((self.n_actions,))

        for i in range(self.n_actions):
            return_val[i] = accessed_dict[self._action_names[i]]

        self._on_access_listener(state_id, apply_get=True)

        return return_val

    def get_state_pins(self):
        return_val = ''
        for i in range(self.state_length):
            state_pin_i = self._states_pins[i]
            return_val += f"State No {i}, has {len(state_pin_i)} pins: {state_pin_i}\n"
        return return_val.strip()
    
    def export2pandas(self):
        df = pd.DataFrame(self._q_table.values())
        df.to_csv(self.csv_export_path, index=False)
        return df
        
