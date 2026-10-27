import numpy as np
import cv2

_X = 1
_O = 2
_N = 0

_CONTINUE = 0
_TIE = 1
_WINS = 2
_ILLEGAL = 3

_gridShape = (3, 3)

class XO_Env:
    def __init__(self):
        self.reset()

    def _conjugatePlayer(self, p):
        return _O if p == _X else _X

    # --------------------------------------------------
    #  Info
    # --------------------------------------------------
    def getStatesActionsNames(self):
        state_names = ["current_player"]
        action_names = []

        for i in range(3):
            for j in range(3):
                state_names.append(f"cell_{i}_{j}")
                action_names.append(f"place_in_{i}_{j}")

        return state_names, action_names

    def getStatesMinMax(self):
        state_min = [1]
        state_max = [2]

        for _ in range(9):
            state_min.append(0)
            state_max.append(2)

        return np.array(state_min), np.array(state_max)

    def _getCurrentState(self):
        s = [self._curPlayer]
        s.extend(self._grid.flatten())
        return np.array(s, dtype=int).flatten()

    def getStateShape(self):
        return self._getCurrentState().shape

    def getActionsCount(self):
        return 9

    def getGrid(self):
        return self._grid.copy()

    # --------------------------------------------------
    #  Reset
    # --------------------------------------------------
    def reset(self):
        self._grid = np.zeros(_gridShape, dtype=int)
        self._curPlayer = np.random.choice([_X, _O])
        return self._getCurrentState()

    # --------------------------------------------------
    #  Rendering
    # --------------------------------------------------
    def render(self):
        dim = 248
        txt = {_X: "X", _O: "O", _N: "-"}
        colors = {_X: (255, 0, 0), _O: (0, 255, 0), _N: (100, 100, 100)}

        img = np.ones((dim, dim, 3), dtype=np.uint8) * 255
        seg = dim / 4
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(3):
            for j in range(3):
                x = int(seg * (j + 0.5) - 10)
                y = int(seg * (i + 0.5) + 10)
                cv2.putText(img, txt[self._grid[i, j]], (x, y), font, 1,
                            colors[self._grid[i, j]], 2)
        return img

    # --------------------------------------------------
    #  Game logic
    # --------------------------------------------------
    def _playerWins(self, p):
        g = self._grid
        # rows
        for i in range(3):
            if (g[i, 0] == g[i, 1] == g[i, 2] == p):
                return True
        # cols
        for j in range(3):
            if (g[0, j] == g[1, j] == g[2, j] == p):
                return True
        # diagonals
        if (g[0, 0] == g[1, 1] == g[2, 2] == p):
            return True
        if (g[0, 2] == g[1, 1] == g[2, 0] == p):
            return True
        return False

    # --------------------------------------------------
    #  Step for single move
    # --------------------------------------------------
    def _one_step(self, action):
        row, col = action // 3, action % 3

        # Illegal move
        if self._grid[row, col] != _N:
            self._gameDone = True
            return self._getCurrentState(), _ILLEGAL, -15

        # Apply move
        self._grid[row, col] = self._curPlayer

        # Check win
        if self._playerWins(self._curPlayer):
            self._gameDone = True
            return self._getCurrentState(), _WINS, 1.0

        # Check tie
        if not (self._grid == _N).any():
            self._gameDone = True
            return self._getCurrentState(), _TIE, 0.5

        # Continue game: swap player + small step penalty
        self._curPlayer = self._conjugatePlayer(self._curPlayer)
        return self._getCurrentState(), _CONTINUE, -0.04

    # --------------------------------------------------
    #  Step for agent + opponent moves
    # --------------------------------------------------
    def two_steps(self, agent_policy, opponent_policy):
        # Agent move
        agent_action = agent_policy(self._getCurrentState())
        state, flag, reward = self._one_step(agent_action)

        if flag == _WINS or flag == _TIE or flag == _ILLEGAL:
            return state, True, reward

        # Opponent move
        opponent_action = opponent_policy(self._getCurrentState())
        state, flag, opp_reward = self._one_step(opponent_action)

        # Opponent wins → negative reward for agent
        if flag == _WINS:
            return state, True, opp_reward * -1
        if flag == _TIE:
            return state, True, opp_reward * 1
        if flag == _ILLEGAL:
            pass

        # Continue game
        return state, False, reward