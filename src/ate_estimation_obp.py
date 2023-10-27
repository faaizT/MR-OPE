# implementing an experiment to evaluate the accuracy of OPE using classification data
from pathlib import Path

# from dowhy import CausalModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import pandas as pd

# import open bandit pipeline (obp)
from obp.ope import (
    RegressionModel,
)
from sklearn.neural_network import MLPClassifier
from obp_utils.custom_dataset import CustomBanditDataset, get_ate_dataset

from obp_utils.ope import create_ope_object
import logging
import argparse

from utils.helper_functions import create_tag, str2bool


"""
This is a tutorial implementation of OPE using the OBP library (https://github.com/st-tech/zr-obp).
Here, we use the library to run OPE experiments for IPW, DM and DR estimators (and other baselines) using synthetic data.
"""


def set_args():
    parser = argparse.ArgumentParser(description="Parser for the OPE experiments")
    parser.add_argument(
        "--results_dir", default=".", type=str, help="Folder to save the results"
    )
    parser.add_argument(
        "--n_train", type=int, default=5000, help="Number of training points for model"
    )
    parser.add_argument(
        "--n_eval", type=int, default=100, help="Number of evaluation points for OPE"
    )
    parser.add_argument(
        "--balance_param",
        type=float,
        default=0.9,
        help="Proportion of datapoints with A=1 in observational dataset",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="continuous",
        help="Whether the reward is continuous or binary",
    )
    parser.add_argument("--seed", type=int, default=1095, help="Seed for the run")
    parser.add_argument(
        "--use_train_behav",
        type=str2bool,
        default="True",
        help="Approximate behaviour policy",
    )
    parser.add_argument(
        "--estimation_method",
        type=str,
        default="default",
        help="Which estimation method to use to compute MR",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="twins",
        help="Which ate dataset to use",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="/data/ziz/not-backed-up/taufiq/twins-dataset/twins-data.csv",
        help="Where preprocessed data is saved",
    )
    args = parser.parse_args()
    return args


def main(args):
    n_train = args.n_train
    use_train_behav = args.use_train_behav
    np.random.seed(args.seed)
    dataset = get_ate_dataset(args.dataset, args.dataset_file, args.balance_param)
    reward_type = dataset.reward_type
    if args.n_eval > len(dataset.data):
        return
    bandit_feedback_train, bandit_feedback_test = dataset.obtain_batch_bandit_feedback(
        test_size=args.n_eval
    )
    logging.info("Data generated")
    # n_train = bandit_feedback_train["action"].shape[0]

    # (3) Off-Policy Evaluation
    if reward_type == "binary":
        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=RandomForestClassifier(
                n_estimators=10,
                max_samples=0.8,
                max_depth=2,
                random_state=args.seed,
            ),
        )
    else:
        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=LinearRegression(),
        )
    logging.info("Fitting a rewards model")
    regression_model.fit(
        context=bandit_feedback_train["context"][: n_train // 2],
        action=bandit_feedback_train["action"][: n_train // 2],
        reward=bandit_feedback_train["reward"][: n_train // 2],
    )

    logging.info("Rewards model fitting completed")
    estimated_rewards_by_reg_model = regression_model.predict(
        context=bandit_feedback_test["context"]
    )
    ground_truth = dataset.ground_truth_ate
    logging.info(f"Ground truth policy value: {ground_truth}")

    if use_train_behav:
        logging.info("Fitting behaviour policy")
        # Use half the training data to train the behaviour policy
        behav_policy = RandomForestClassifier(
            n_estimators=100,
            max_samples=0.8,
            max_depth=4,
            random_state=args.seed,
        )
        behav_policy.fit(
            X=bandit_feedback_train["context"][:n_train],
            y=bandit_feedback_train["action"][:n_train],
        )
        pi_b_all_actions_training = behav_policy.predict_proba(
            X=bandit_feedback_train["context"]
        )
        pi_b_all_actions_testing = behav_policy.predict_proba(
            X=bandit_feedback_test["context"]
        )
        pi_b_scores_train = pi_b_all_actions_training[
            np.arange(pi_b_all_actions_training.shape[0]),
            bandit_feedback_train["action"],
        ]
        pi_b_scores_test = pi_b_all_actions_testing[
            np.arange(pi_b_all_actions_testing.shape[0]),
            bandit_feedback_test["action"],
        ]
    else:
        pi_b_scores_test = bandit_feedback_test["pscore"]
        pi_b_scores_train = bandit_feedback_train["pscore"]

    # Use the other half of training data to train the weights for Marginal ratios
    # First we estimate E[Y(0)]
    ope0 = create_ope_object(
        bandit_feedback_test=bandit_feedback_test,
        use_train_behav=use_train_behav,
        pi_b_scores=pi_b_scores_train[:n_train],
        pi_e_scores=bandit_feedback_train["action"][:n_train] == 0,
        reward=bandit_feedback_train["reward"][:n_train],
        reward_type=reward_type,
        estimation_method=args.estimation_method,
    )

    action_dist0 = np.zeros((bandit_feedback_test["context"].shape[0], 2, 1))
    action_dist0[:, 0, 0] = 1

    action_dist1 = np.zeros((bandit_feedback_test["context"].shape[0], 2, 1))
    action_dist1[:, 1, 0] = 1

    policy_value_ey_0 = ope0.estimate_policy_values(
        estimated_pscore=pi_b_scores_test,
        action_dist=action_dist0,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    logging.info("Relative estimation errors for policy value:")
    logging.info("\n" + str(policy_value_ey_0))

    # Next we estimate E[Y(1)]
    ope1 = create_ope_object(
        bandit_feedback_test=bandit_feedback_test,
        use_train_behav=use_train_behav,
        pi_b_scores=pi_b_scores_train[:n_train],
        pi_e_scores=bandit_feedback_train["action"][:n_train] == 1,
        reward=bandit_feedback_train["reward"][:n_train],
        reward_type=reward_type,
        estimation_method=args.estimation_method,
    )

    policy_value_ey_1 = ope1.estimate_policy_values(
        estimated_pscore=pi_b_scores_test,
        action_dist=action_dist1,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    logging.info("Relative estimation errors for policy value:")
    logging.info("\n" + str(policy_value_ey_1))

    results_df = (
        (
            pd.DataFrame(policy_value_ey_1, index=[0])
            - pd.DataFrame(policy_value_ey_0, index=[0])
        )
        .transpose()
        .reset_index()
    )
    results_df.rename(columns={0: "ate_value"}, inplace=True)
    results_df["ground_truth_value"] = ground_truth
    results_df["bias"] = results_df["ate_value"] - results_df["ground_truth_value"]
    logging.info("ATE results:")
    logging.info("\n" + str(results_df))

    tag = create_tag(args, ignore_cols=["results_dir", "dataset_file"])
    args_df = pd.DataFrame(vars(args), index=results_df.index)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    pd.concat([results_df, args_df], axis=1).reset_index(drop=True).to_csv(
        f"{args.results_dir}/results_{tag}.csv", index=False
    )
    logging.info("Results saved")


if __name__ == "__main__":
    args = set_args()
    main(args)
    # (1) Generate Synthetic Bandit Data
