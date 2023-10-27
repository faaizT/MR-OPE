import argparse
import numpy as np
import torch
import math


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_tag(args, ignore_cols=[]):
    args_dict = vars(args)
    tag = ""
    for arg in args_dict:
        if arg not in ignore_cols:
            tag += f"_{arg}{args_dict[arg]}"
    return tag


def ipw_estimator(X, a, Y, behav_policy_model):
    return (
        Y[a == 1].reshape(-1) / behav_policy_model(X)[(a == 1).reshape(-1), 1]
    ).sum() / Y.shape[0] - (
        Y[a == 0].reshape(-1) / behav_policy_model(X)[(a == 0).reshape(-1), 0]
    ).sum() / Y.shape[
        0
    ]


def ipw_estimator_cts(X, a, Y, behav_policy_model, target_policy):
    policy_ratio = (
        target_policy.density_value(X.reshape(-1), a.reshape(-1))
        / behav_policy_model.density_value(X.reshape(-1), a.reshape(-1))
    ).reshape(-1, 1)
    return (policy_ratio * Y).sum() / Y.shape[0]


def aipw_estimator(X, a, Y, behav_policy_model, e_y_0_model, e_y_1_model):
    return (
        ipw_estimator(X, a, Y, behav_policy_model)
        - (
            (a.reshape(-1) - behav_policy_model(X)[:, 1])
            * e_y_1_model(X).reshape(-1)
            / behav_policy_model(X)[:, 1]
        ).sum()
        / Y.shape[0]
        - (
            (a.reshape(-1) - behav_policy_model(X)[:, 1])
            * e_y_0_model(X).reshape(-1)
            / behav_policy_model(X)[:, 0]
        ).sum()
        / Y.shape[0]
    )


def aipw_estimator_cts(X, a, Y, behav_policy_model, target_policy, ey_xa_model, n=100):
    with torch.no_grad():
        target_actions = target_policy.sample(X.repeat(1, n).reshape(-1)).reshape(-1, 1)
        direct_method = ey_xa_model(
            torch.cat([X.repeat(1, n).reshape(-1, 1), target_actions], -1)
        ).mean()
        mu_x_a = ey_xa_model(torch.cat([X, a], -1)).reshape(-1)
    return (
        ipw_estimator_cts(
            X, a, Y.reshape(-1) - mu_x_a, behav_policy_model, target_policy
        )
        + direct_method
    )


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        try:
            return (
                1
                / torch.sqrt(2 * np.pi * self.sigma**2)
                * torch.exp(-((x.reshape(-1) - self.mu) ** 2) / (2 * self.sigma**2))
            ).reshape(-1, 1)
        except:
            return (
                1
                / math.sqrt(2 * np.pi * self.sigma**2)
                * torch.exp(-((x.reshape(-1) - self.mu) ** 2) / (2 * self.sigma**2))
            ).reshape(-1, 1)
