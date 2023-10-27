import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


class GeneralisedMIPS:
    def __init__(self, embeddings, pi_b, pi_e):
        self._fit_w_r(embeddings, pi_b, pi_e)

    def _fit_w_r(self, embeddings, pi_b, pi_e):
        self.w_r_nn = LinearRegression()
        policy_ratios = pi_e / pi_b
        self.policy_ratios = policy_ratios.reshape(-1)
        self.embeddings = embeddings
        logging.info(
            f"Range of policy ratios: {policy_ratios.min()}-{policy_ratios.max()}"
        )
        policy_ratios_normalised = (policy_ratios - policy_ratios.mean()) / max(
            policy_ratios.std(), 1
        )
        reward_normalised = (embeddings - embeddings.mean(0)) / embeddings.std(0)
        self.w_r_nn.fit(
            X=reward_normalised,
            y=policy_ratios_normalised.reshape(-1, 1),
        )
        self.w_r = (
            lambda x: self.w_r_nn.predict(
                (x - embeddings.mean(0)) / embeddings.std(0)
            ).reshape(-1)
            * max(policy_ratios.std(), 1)
            + policy_ratios.mean()
        )
        self.w_r_weights = self.w_r(embeddings).reshape(-1)
        logging.info(
            f"Range of ratios w(r): {self.w_r_weights.min()}-{self.w_r_weights.max()}"
        )

    def estimate_policy_value(self, embeddings, reward):
        return (reward * self.w_r(embeddings)).mean()
