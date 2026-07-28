import os
import random
import pickle
import numpy as np

import gymnasium as gym


# Fine tune parameters
PROFIT_SCALE = 10000
N_BAD_MONTHS = 24
N_SEEDS = 20
EVALUATION_SEEDS = list(range(20, 40))


class BaselineABMEnv(gym.Env):
    """
    One RL step = one ABM month. Reward = real monthly profit
    Terminal (bankruptcy or 24 months zero profits): Φ(terminal) = 0 to preserve policy invariance.
    """

    def __init__(self, seed=None, model=None):
        # Initialize the model (pick a random snapshot)
        self._load_model(seed=seed, model=model)

        self.action_space = gym.spaces.Dict({
            'hr_action': gym.spaces.Discrete(3),
            'price_adjustment': gym.spaces.Box(low=-0.02, high=0.02),
            'wage_adjustment': gym.spaces.Box(low=-0.02, high=0.02)
        })

        self.observation_space = gym.spaces.Dict({
            'open_position': gym.spaces.Discrete(2),
            'months_since_hire_failure': gym.spaces.Box(low=0, high=np.inf, dtype=np.int32),
            'employee_count': gym.spaces.Box(low=0, high=len(self._model.households), dtype=np.int16),
            'liquidity_buffer': gym.spaces.Box(low=0, high=np.inf, dtype=np.float32),
            'inventories': gym.spaces.Box(low=0, high=np.inf, dtype=np.float32),
            'wage_rate': gym.spaces.Box(low=0, high=np.inf, dtype=np.float32),
            'goods_price': gym.spaces.Box(low=0, high=np.inf, dtype=np.float32),
            'last_month_demand': gym.spaces.Box(low=0, high=np.inf, dtype=np.float32)
        })

    def step(self, action):
        # Apply the monthly decision, then simulate one month (21 days)
        firm = self._model.learning_firms[0]
        firm.apply_action(action)

        for _ in range(self._model.parameters.month_length):
            self._model.step()

        reward = self._compute_reward()
        observation = self._get_obs()
        info = self._get_info()

        # Termination: bankruptcy or long zero-profit streak
        terminated = firm.is_bankrupt or firm.zero_profit_streak >= N_BAD_MONTHS
        truncated = False # firm.zero_profit_streak >= N_BAD_MONTHS
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._load_model(seed=seed, model=options.get('model') if options else None)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    # ---------- internals ----------

    def _load_model(self, seed=None, model=None):
        # Use the model if given
        if model is not None:
            self._model = model
            return
        # Else check the seed
        if seed is None:
            seed = random.randint(0, N_SEEDS - 1)
        elif seed not in EVALUATION_SEEDS:
            seed = seed % N_SEEDS
        with open(f'../snapshots/abm_{seed}.pkl', 'rb') as f:
            # Uncomment for debugging
            # print(f'Loading seed {seed}')
            self._model = pickle.load(f)

    def _cpi(self) -> float:
        """Demand-weighted price index using last month's quantities; falls back to simple average."""
        prices = [f.goods_price for f in self._model.firms]
        qtys = [max(0.0, f.last_month_demand) for f in self._model.firms]
        qsum = float(sum(qtys))
        if qsum > 0.0:
            return float(sum(p * q for p, q in zip(prices, qtys)) / qsum)
        # fallback: simple average (should be rare early on)
        return float(sum(prices) / max(1, len(prices)))


    def _compute_reward(self) -> float:
        firm = self._model.learning_firms[0]

        # Base signal: REAL monthly profit (deflated by CPI), normalized & softly clipped
        cpi = max(1e-8, self._cpi())
        profit_real = float(firm.last_month_profit) / cpi
        reward = float(np.clip(profit_real / PROFIT_SCALE, -5.0, 5.0))
        return float(reward)

    def _get_obs(self):
        firm = self._model.learning_firms[0]
        return {
            'open_position': int(firm.open_position),
            'months_since_hire_failure': firm.months_since_hire_failure,
            'employee_count': firm.employee_count,
            'liquidity_buffer': firm.liquidity_buffer,
            'inventories': firm.inventories,
            'wage_rate': firm.wage_rate,
            'goods_price': firm.goods_price,
            'last_month_demand': firm.last_month_demand
        }

    def _get_info(self):
        return {
            'profit_ranking': sorted(
                self._model.firms, key=lambda f: f.last_month_profit, reverse=True
            ).index(self._model.learning_firms[0]) + 1
        }


gym.register(id='BaselineABMEnv', entry_point=BaselineABMEnv)

def make_sb_env():
    # Wrap the environment for stable-baselines training
    # SB doesn't work with Dict action so need to convert it to Box instead
    
    def box_to_dict(action):
        return {
            'hr_action': int(np.clip(np.round(action[0]), 0, 2)),
            'price_adjustment': action[1],
            'wage_adjustment': action[2]
        }
        
    env = BaselineABMEnv()
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.TransformAction(env,
                                       func=box_to_dict,
                                       action_space=gym.spaces.Box(low=np.array([0, -0.02, -0.02]),
                                                                   high=np.array([2, 0.02, 0.02]),
                                                                   dtype=np.float32)
    )
    return env

def make_custom_env():
    def thunk():
        def clip_action(action):
            return {
                'hr_action': action['hr_action'],
                'price_adjustment': np.clip(action['price_adjustment'],
                                            env.action_space['price_adjustment'].low,
                                            env.action_space['price_adjustment'].high),
                'wage_adjustment': np.clip(action['wage_adjustment'],
                                           env.action_space['wage_adjustment'].low,
                                           env.action_space['wage_adjustment'].high)
            }
        
        env = BaselineABMEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.TransformAction(env, clip_action, env.action_space)
        return env

    return thunk

def make_custom_unbounded_env(survival_bonus=None, time_limit=None):
    def thunk():
        env = BaselineABMEnv()
        env = gym.wrappers.FlattenObservation(env)
        if survival_bonus is not None:
            class SurvivalBonus(gym.RewardWrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self._bonus = float(survival_bonus)
                    
                def step(self, action):
                    obs, r, terminated, truncated, info = self.env.step(action)
                    if not terminated:
                        r = r + self._bonus
                    return obs, r, terminated, truncated, info
            env = SurvivalBonus(env)

        if time_limit is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)

        return env

    return thunk