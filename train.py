"""
ML Arena — Chess SB3 Self-Play Training Script
Environment: PettingZoo chess_v6 (wrapped as single-agent Gym env)

Install dependencies:
    pip install -r requirements.txt

Train then upload:
    python run.py

Switching algorithms:
    Edit model.py — change ALGORITHM (must support action masking, e.g. MaskablePPO).
    Re-run train.py and run.py.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import gymnasium as gym
from pettingzoo.classic import chess_v6
from stable_baselines3.common.vec_env import DummyVecEnv

from model import ALGORITHM, POLICY, POLICY_KWARGS, SAVE_PATH

# ═══ ✅ Tune freely: training hyperparameters ══════════════════════
TOTAL_TIMESTEPS = 2_000_000   # recommended: 2M+ for meaningful performance
N_ENVS          = 4         # parallel self-play environments
# ══════════════════════════════════════════════════════════════════


class ChessSelfPlayEnv(gym.Env):
    """Single-agent Gym wrapper for chess_v6.

    The learning agent always plays as the first agent to move (white).
    Opponent moves are selected randomly from legal actions.
    Supports action_masks() for MaskablePPO.
    """

    def __init__(self):
        super().__init__()
        self._env = chess_v6.env()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(8 * 8 * 111,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4672)
        self._action_mask = np.ones(4672, dtype=np.int8)
        self._learning_agent: str = ""

        from model import ALGORITHM, SAVE_PATH
        import os
        if os.path.exists(SAVE_PATH + ".zip"):
            self._opponent = ALGORITHM.load(SAVE_PATH)
        else:
            self._opponent = None
        self._total_steps = 0

    # ═══ ✅ Reward shaping (change freely) ═══════════════════════════
    # PettingZoo chess rewards: win=+1, loss=-1, draw=0 (terminal only).
    # You can add intermediate rewards here, e.g. based on material count.
    def _shape_reward(self, reward: float) -> float:
        # ── 範例 1：子力差獎勵（取消注釋啟用）─────────────────────────
        import chess as _chess
        board = self._env.env.board
        _vals = {_chess.PAWN:1, _chess.KNIGHT:3, _chess.BISHOP:3,
                 _chess.ROOK:5, _chess.QUEEN:9}
        our_color = _chess.WHITE if self._learning_agent == "player_0" else _chess.BLACK
        mat = sum(_vals.get(p.piece_type, 0)
                  for p in board.piece_map().values() if p.color == our_color) \
            - sum(_vals.get(p.piece_type, 0)
                  for p in board.piece_map().values() if p.color != our_color)
        reward += mat * 0.001   # 小額中間獎勵，避免蓋過終局 ±1
        # ──────────────────────────────────────────────────────────────
        # ── 範例 2：每步存活小獎勵 ────────────────────────────────────
        reward += 0.001   # 鼓勵撐住，不要輕易被將死
        # ──────────────────────────────────────────────────────────────
        return reward
    # ══════════════════════════════════════════════════════════════════

    def action_masks(self) -> np.ndarray:
        return self._action_mask.astype(bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset(seed=seed)
        self._learning_agent = self._env.agent_selection
        obs, _, _, _, _ = self._env.last()
        self._action_mask = obs["action_mask"].copy()
        return obs["observation"].flatten().astype(np.float32), {}

    def _is_done(self) -> bool:
        return (
            self._env.terminations.get(self._learning_agent, False)
            or self._env.truncations.get(self._learning_agent, False)
        )

    def step(self, action: int):
        # Ensure the action is legal
        if not self._action_mask[int(action)]:
            legal = np.where(self._action_mask)[0]
            action = int(np.random.choice(legal)) if len(legal) else 0

        self._env.step(int(action))

        # Let opponent(s) play until it is our turn or the game ends
        for _ in range(500):
            if self._is_done():
                reward = self._shape_reward(
                    float(self._env.rewards.get(self._learning_agent, 0.0))
                )
                self._action_mask = np.ones(4672, dtype=np.int8)
                return np.zeros(8 * 8 * 111, dtype=np.float32), reward, True, False, {}

            if self._env.agent_selection == self._learning_agent:
                break

            opp_obs, _, opp_term, opp_trunc, _ = self._env.last()
            if opp_term or opp_trunc:
                self._env.step(None)
                continue
            opp_mask = opp_obs["action_mask"]

            # ── 【改善版】Self-Play 實作與動態更新 ──
            self._total_steps += 1
            
            # 每 10 萬步重新載入最新模型以更新對手
            if self._total_steps % 100_000 == 0:
                from model import ALGORITHM, SAVE_PATH
                import os
                if os.path.exists(SAVE_PATH + ".zip") or os.path.exists(SAVE_PATH):
                    self._opponent = ALGORITHM.load(SAVE_PATH)

            # 若對手模型存在則進行推論（Self-Play），否則採取隨機動作
            if getattr(self, "_opponent", None) is not None:
                opp_flat = opp_obs["observation"].flatten().astype(np.float32)
                action, _ = self._opponent.predict(opp_flat,
                                                   action_masks=opp_mask.astype(bool),
                                                   deterministic=False)
                self._env.step(int(action))
                continue
            else:
                legal = np.where(opp_mask)[0]
                self._env.step(int(np.random.choice(legal)) if len(legal) else 0)

        if self._is_done():
            reward = self._shape_reward(
                float(self._env.rewards.get(self._learning_agent, 0.0))
            )
            self._action_mask = np.ones(4672, dtype=np.int8)
            return np.zeros(8 * 8 * 111, dtype=np.float32), reward, True, False, {}

        obs, _, term, trunc, info = self._env.last()
        self._action_mask = obs["action_mask"].copy()
        reward = self._shape_reward(
            float(self._env.rewards.get(self._learning_agent, 0.0))
        )
        return obs["observation"].flatten().astype(np.float32), reward, term, trunc, info

    def close(self):
        self._env.close()


def main():
    env = DummyVecEnv([ChessSelfPlayEnv for _ in range(N_ENVS)])

    model = ALGORITHM(
        POLICY,
        env,
        policy_kwargs=POLICY_KWARGS or None,
        verbose=1,
    )

    print(f"Training {ALGORITHM.__name__} for {TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(SAVE_PATH)
    print(f"\nModel saved as {SAVE_PATH}.zip — ready to upload with run.py")
    env.close()


if __name__ == "__main__":
    main()
