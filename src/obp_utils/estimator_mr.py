from dataclasses import dataclass
import itertools
from typing import Dict
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.base import is_classifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_scalar
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from obp.utils import check_array
from obp.utils import check_ope_inputs
from obp.ope import InverseProbabilityWeighting
import logging


@dataclass
class MarginalizedRatioWeighting(InverseProbabilityWeighting):
    def __init__(
        self,
        reward,
        pi_b,
        pi_e,
        reward_type,
        estimation_method="default",
        use_estimated_pscore=False,
        estimator_name="mr",
        **kwargs,
    ):
        self.estimator_name = estimator_name
        self.estimation_method = estimation_method
        self.reward_type = reward_type
        self.use_estimated_pscore = use_estimated_pscore
        if reward_type == "binary":
            self._fit_w_y_binary(reward, pi_b, pi_e)
        elif estimation_method == "default":
            self._fit_w_y(reward, pi_b, pi_e)
        else:
            self._fit_w_y_alternative(reward, pi_b, pi_e)

    def _fit_w_y(self, reward, pi_b, pi_e):
        self.w_y_nn = MLPRegressor(
            hidden_layer_sizes=(512, 256, 32),
            verbose=True,
            alpha=0.01,
            max_iter=1000,
            early_stopping=False,
        )
        policy_ratios = pi_e / pi_b
        self.policy_ratios = policy_ratios.reshape(-1)
        self.reward = reward.reshape(-1)
        logging.info(
            f"Range of policy ratios: {policy_ratios.min()}-{policy_ratios.max()}"
        )
        policy_ratios_normalised = (policy_ratios - policy_ratios.mean()) / max(
            policy_ratios.std(), 1
        )
        reward_normalised = (reward - reward.mean()) / reward.std()
        self.w_y_nn.fit(
            X=reward_normalised.reshape(-1, 1),
            y=policy_ratios_normalised.reshape(-1, 1),
        )
        self.w_y = (
            lambda x: self.w_y_nn.predict(
                (x.reshape(-1, 1) - reward.mean()) / reward.std()
            ).reshape(-1)
            * max(policy_ratios.std(), 1)
            + policy_ratios.mean()
        )
        self.w_y_weights = self.w_y(reward).reshape(-1)
        logging.info(
            f"Range of ratios w(y): {self.w_y_weights.min()}-{self.w_y_weights.max()}"
        )

    def _fit_w_y_alternative(self, reward, pi_b, pi_e):
        self.w_y_nn = MLPRegressor(
            hidden_layer_sizes=(512, 256, 32),
            verbose=True,
            alpha=0.01,
            max_iter=1000,
            early_stopping=False,
        )
        policy_ratios = pi_e / pi_b
        logging.info(
            f"Range of policy ratios: {policy_ratios.min()}-{policy_ratios.max()}"
        )
        target = reward * pi_e / pi_b
        target_normalised = (target - target.mean()) / max(target.std(), 1)
        reward_normalised = (reward - reward.mean()) / reward.std()
        self.w_y_nn.fit(
            X=reward_normalised.reshape(-1, 1),
            y=target_normalised.reshape(-1, 1),
        )
        self.h_y = (
            lambda x: self.w_y_nn.predict(
                (x.reshape(-1, 1) - reward.mean()) / reward.std()
            ).reshape(-1)
            * max(target.std(), 1)
            + target.mean()
        )
        h_y = self.h_y(reward).reshape(-1)
        logging.info(f"Range of h_values h(y): {h_y.min()}-{h_y.max()}")

    def _fit_w_y_binary(self, reward, pi_b, pi_e):
        policy_ratios = pi_e / pi_b
        self.policy_ratios = policy_ratios.reshape(-1)
        e_policy_ratios_y0 = policy_ratios[reward == 0].mean()
        e_policy_ratios_y1 = policy_ratios[reward == 1].mean()
        self.w_y = (
            lambda x: (x == 0) * e_policy_ratios_y0 + (x == 1) * e_policy_ratios_y1
        )
        self.w_y_weights = self.w_y(reward).reshape(-1)
        self.reward = reward.reshape(-1)

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action: np.ndarray,
        pscore: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if self.estimation_method == "alternative" and self.reward_type == "continuous":
            return self.h_y(reward)
        return reward * self.w_y(reward)

    def save_ratio_values(self, file_path: str):
        if self.estimation_method == "alternative" and self.reward_type == "continuous":
            raise NotImplementedError("Function not Implemented for alternative method")
        df = pd.DataFrame(
            data={
                "policy_ratios": self.policy_ratios,
                "weights_w_y": self.w_y_weights,
                "y": self.reward,
            }
        )
        df.to_csv(file_path, index=False)
