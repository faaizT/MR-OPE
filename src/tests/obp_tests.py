import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import unittest
import numpy as np
from obp.dataset import SyntheticBanditDataset
from obp.ope import (
    OffPolicyEvaluation,
)
from obp_utils.estimator_mr import MarginalizedRatioWeighting as MR


class TestOBPImplementation(unittest.TestCase):
    def test_mr_implementation_for_discrete_rewards_synthetic_dataset(self):
        # Generate data
        reward_type = "binary"
        dataset = SyntheticBanditDataset(
            n_actions=10,
            reward_type=reward_type,
        )
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=10000)
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=10000)
        # The target policy is just a random policy
        action_dist = (
            np.ones((bandit_feedback_test["reward"].shape[0], dataset.n_actions, 1))
            / dataset.n_actions
        )
        pi_e_scores_all_actions = (
            np.ones((bandit_feedback_train["reward"].shape[0], dataset.n_actions))
            / dataset.n_actions
        )
        pi_e_scores = pi_e_scores_all_actions[
            np.arange(pi_e_scores_all_actions.shape[0]),
            bandit_feedback_train["action"],
        ]
        # Define the MR estimator
        mr_estimator = MR(
            reward=bandit_feedback_train["reward"],
            pi_b=bandit_feedback_train["pscore"],
            pi_e=pi_e_scores,
            reward_type=reward_type,
        )
        w_y = mr_estimator.w_y
        # Calculate the MR policy value
        mr_policy_value = (
            bandit_feedback_test["reward"] * w_y(bandit_feedback_test["reward"])
        ).mean()
        # Calculate the MR policy value using OBP policy
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback_test,
            ope_estimators=[
                mr_estimator,
            ],
        )
        (
            estimated_policy_value_a,
            estimated_interval_a,
        ) = ope.summarize_off_policy_estimates(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=np.ones_like(action_dist),
        )
        mr_policy_value_obp = estimated_policy_value_a["estimated_policy_value"]["mr"]
        # Ensure that the two policy values are identical
        self.assertEqual(mr_policy_value - mr_policy_value_obp, 0.0)

    def test_mr_implementation_for_continuous_rewards_synthetic_dataset(self):
        # Generate data
        reward_type = "continuous"
        dataset = SyntheticBanditDataset(
            n_actions=10,
            reward_type=reward_type,
        )
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=10000)
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=10000)
        # The target policy is just a random policy
        action_dist = (
            np.ones((bandit_feedback_test["reward"].shape[0], dataset.n_actions, 1))
            / dataset.n_actions
        )
        pi_e_scores_all_actions = (
            np.ones((bandit_feedback_train["reward"].shape[0], dataset.n_actions))
            / dataset.n_actions
        )
        pi_e_scores = pi_e_scores_all_actions[
            np.arange(pi_e_scores_all_actions.shape[0]),
            bandit_feedback_train["action"],
        ]
        # Define the MR estimator
        mr_estimator = MR(
            reward=bandit_feedback_train["reward"],
            pi_b=bandit_feedback_train["pscore"],
            pi_e=pi_e_scores,
            reward_type=reward_type,
        )
        w_y = mr_estimator.w_y
        # Calculate the MR policy value
        mr_policy_value = (
            bandit_feedback_test["reward"] * w_y(bandit_feedback_test["reward"])
        ).mean()
        # Calculate the MR policy value using OBP policy
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback_test,
            ope_estimators=[
                mr_estimator,
            ],
        )
        (
            estimated_policy_value_a,
            estimated_interval_a,
        ) = ope.summarize_off_policy_estimates(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=np.ones_like(action_dist),
        )
        mr_policy_value_obp = estimated_policy_value_a["estimated_policy_value"]["mr"]
        # Ensure that the two policy values are identical
        self.assertEqual(mr_policy_value - mr_policy_value_obp, 0.0)


if __name__ == "__main__":
    unittest.main()
