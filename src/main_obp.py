# implementing an experiment to evaluate the accuracy of OPE using classification data
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as np
import pandas as pd

# import open bandit pipeline (obp)
from obp.policy import IPWLearner
from obp.dataset import (
    SyntheticBanditDataset,
    linear_reward_function,
    logistic_reward_function,
    polynomial_reward_function,
    quadratic_reward_funcion_continuous,
)
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting as IPW,
    DirectMethod as DM,
    DoublyRobust as DR,
)
from obp_utils.estimator_gmips import GeneralisedMIPS
from obp_utils.estimator_mr import MarginalizedRatioWeighting as MR
from obp_utils.model_wrapper import PolicyWrapper, PolicyWrapperAlpha
from obp_utils.rewards import (
    reward_indep_action,
    reward_context_times_action,
    reward_binary,
    reward_context_norm_plus_action,
    reward_context_sin_norm,
    reward_sin_ax,
    reward_ax_first_5_components,
    reward_sine_ax_first_5_components,
    reward_sin_ax_plus_one,
    reward_sin_ax_normalised,
)
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
        "--n_train", type=int, default=1000, help="Number of training points for model"
    )
    parser.add_argument(
        "--n_eval", type=int, default=100, help="Number of evaluation points for OPE"
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="continuous",
        help="Whether the reward is continuous or binary",
    )
    parser.add_argument(
        "--estimation_method",
        type=str,
        default="default",
        help="Which estimation method to use to compute MR",
    )
    parser.add_argument("--seed", type=int, default=3, help="Seed for the run")
    parser.add_argument(
        "--beta_behav",
        type=float,
        default=1,
        help="Beta parameter for the behavioural policy",
    )
    parser.add_argument(
        "--alpha_target",
        type=float,
        default=0.8,
        help="Alpha parameter for the target policy",
    )
    parser.add_argument(
        "--use_train_behav",
        type=str2bool,
        default="True",
        help="Approximate behaviour policy",
    )
    parser.add_argument(
        "--save_ratios",
        type=str2bool,
        default="False",
        help="save_ratios",
    )
    parser.add_argument(
        "--include_gmips",
        type=str2bool,
        default="True",
        help="Include GMIPS among baselines (only implemented for sin(a.||x||)) reward",
    )
    parser.add_argument("--n_actions", type=int, default=50, help="Number of actions")
    parser.add_argument(
        "--dim_context", type=int, default=5000, help="Context dimension"
    )
    parser.add_argument(
        "--reward_std",
        type=float,
        default=0.01,
        help="Standard deviation of p(Y|X, A)",
    )
    args = parser.parse_args()
    return args


def main(args):
    reward_type = args.reward_type
    n_eval = args.n_eval
    n_train = args.n_train
    beta_behav = args.beta_behav
    use_train_behav = args.use_train_behav
    np.random.seed(args.seed)
    if reward_type == "continuous":
        reward_function = reward_sin_ax
    else:
        reward_function = reward_sin_ax_plus_one
    dataset = SyntheticBanditDataset(
        random_state=args.seed,
        n_actions=args.n_actions,
        reward_type=reward_type,
        dim_context=args.dim_context,
        reward_std=args.reward_std,
        beta=beta_behav,
        reward_function=reward_function,
    )
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_train)
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_eval)
    logging.info("Data generated")

    # (2) Off-Policy Learning
    # Use half the training data to construct an evaluation policy
    logging.info("Fitting an evaluation policy")
    eval_policy = LogisticRegression(verbose=True)
    eval_policy.fit(
        X=bandit_feedback_train["context"][: n_train // 2],
        y=bandit_feedback_train["action"][: n_train // 2],
    )
    eval_policy = PolicyWrapperAlpha(eval_policy, args.alpha_target, args.n_actions)
    pi_e_all_actions_training = eval_policy.predict_proba(
        X=bandit_feedback_train["context"]
    )
    pi_e_scores_train = pi_e_all_actions_training[
        np.arange(pi_e_all_actions_training.shape[0]),
        bandit_feedback_train["action"],
    ]
    action_dist = np.expand_dims(
        eval_policy.predict_proba(X=bandit_feedback_test["context"]), axis=-1
    )
    pi_e_scores_test = action_dist[
        np.arange(action_dist.shape[0]), bandit_feedback_test["action"], 0
    ]
    logging.info("Evaluation policy fitting completed")

    # (3) Off-Policy Evaluation
    if reward_type == "binary":
        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=MLPClassifier(max_iter=10, hidden_layer_sizes=(1,)),
        )
    else:
        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            base_model=RandomForestRegressor(
                n_estimators=10,
                max_samples=0.8,
                max_depth=2,
                random_state=args.seed,
            ),
        )
    logging.info("Fitting a rewards model")
    regression_model.fit(
        context=bandit_feedback_train["context"][:n_train],
        action=bandit_feedback_train["action"][:n_train],
        reward=bandit_feedback_train["reward"][:n_train],
    )

    logging.info("Rewards model fitting completed")
    estimated_rewards_by_reg_model = regression_model.predict(
        context=bandit_feedback_test["context"]
    )
    action_dist_full = np.expand_dims(
        eval_policy.predict_proba(
            X=np.concatenate(
                [bandit_feedback_train["context"], bandit_feedback_test["context"]],
                axis=0,
            )
        ),
        axis=-1,
    )
    expected_reward_full = np.concatenate(
        [
            bandit_feedback_train["expected_reward"],
            bandit_feedback_test["expected_reward"],
        ],
        axis=0,
    )

    ground_truth = dataset.calc_ground_truth_policy_value(
        expected_reward=expected_reward_full,
        action_dist=action_dist_full,
    )
    logging.info(f"Ground truth policy value: {ground_truth}")

    if use_train_behav:
        logging.info("Fitting behaviour policy")
        # Use half the training data to train the behaviour policy
        # behav_policy = LogisticRegression(max_iter=2000)
        behav_policy = RandomForestClassifier()
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
        pi_b_scores_test = np.clip(pi_b_scores_test, 1e-6, 1 - 1e-6)
        pi_b_scores_train = np.clip(pi_b_scores_train, 1e-6, 1 - 1e-6)
    else:
        pi_b_scores_test = bandit_feedback_test["pscore"]
        pi_b_scores_train = bandit_feedback_train["pscore"]

    # Use the other half of training data to train the weights for Marginal ratios
    ope = create_ope_object(
        bandit_feedback_test=bandit_feedback_test,
        use_train_behav=use_train_behav,
        pi_b_scores=pi_b_scores_train[:n_train],
        pi_e_scores=pi_e_scores_train[:n_train],
        reward=bandit_feedback_train["reward"][:n_train],
        reward_type=reward_type,
        estimation_method=args.estimation_method,
    )

    relative_estimation_errors = ope.summarize_estimators_comparison(
        estimated_pscore=pi_b_scores_test,
        ground_truth_policy_value=ground_truth,
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        metric="relative-ee",
    )
    relative_estimation_errors["ground_truth_value"] = ground_truth
    logging.info("Relative estimation errors for policy value:")
    logging.info("\n" + str(relative_estimation_errors))

    squared_errors = ope.summarize_estimators_comparison(
        estimated_pscore=pi_b_scores_test,
        ground_truth_policy_value=ground_truth,
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        metric="se",
    )
    logging.info("Squared errors for policy value:")
    logging.info("\n" + str(squared_errors))

    estimated_policy_value_a, estimated_interval_a = ope.summarize_off_policy_estimates(
        estimated_pscore=pi_b_scores_test,
        action_dist=action_dist,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
    )
    estimated_interval_a["interval_length"] = (
        estimated_interval_a["95.0% CI (upper)"]
        - estimated_interval_a["95.0% CI (lower)"]
    )
    estimated_policy_value_a["bias"] = (
        estimated_policy_value_a["estimated_policy_value"] - ground_truth
    )
    logging.info("CIs for policy value:")
    logging.info("\n" + str(estimated_interval_a))

    logging.info("Policy values:")
    logging.info("\n" + str(estimated_policy_value_a))

    results_df = pd.concat(
        [
            relative_estimation_errors,
            squared_errors,
            estimated_policy_value_a,
            estimated_interval_a,
        ],
        axis=1,
    )

    tag = create_tag(args, ignore_cols=["results_dir", "save_ratios"])
    args_df = pd.DataFrame(vars(args), index=results_df.index)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    pd.concat([results_df, args_df], axis=1).reset_index(drop=False).to_csv(
        f"{args.results_dir}/results_{tag}.csv", index=False
    )
    logging.info("Results saved")

    # Generalised MIPS estimator
    if args.include_gmips:
        logging.info("Calculating GMIPS estimate")
        embeddings_train = np.concatenate(
            [
                np.linalg.norm(
                    bandit_feedback_train["context"][n_train // 2 :], axis=-1
                ).reshape(-1, 1),
                bandit_feedback_train["action"][n_train // 2 :].reshape(-1, 1),
            ],
            axis=1,
        )

        gmips = GeneralisedMIPS(
            embeddings_train,
            pi_b=pi_b_scores_train[n_train // 2 :],
            pi_e=pi_e_scores_train[n_train // 2 :],
        )

        embeddings_test = np.concatenate(
            [
                np.linalg.norm(bandit_feedback_test["context"], axis=-1).reshape(-1, 1),
                bandit_feedback_test["action"].reshape(-1, 1),
            ],
            axis=1,
        )
        gmips_value = gmips.estimate_policy_value(
            embeddings_test, bandit_feedback_test["reward"]
        )
        mips_df = pd.DataFrame(
            {
                "estimated_policy_value": [gmips_value],
                "ground_truth_value": [ground_truth],
                "bias": [gmips_value - ground_truth],
            },
            index=["gmips"],
        )
        args_df = pd.DataFrame(vars(args), index=mips_df.index)
        pd.concat([mips_df, args_df], axis=1).reset_index(drop=False).to_csv(
            f"{args.results_dir}/gmips_{tag}.csv", index=False
        )
        logging.info("Saved GMIPS results")
    # Save the ratio values for policy ratios as well as our method
    if args.save_ratios:
        ope.ope_estimators[-1].save_ratio_values(
            file_path=f"{args.results_dir}/ratios_{tag}.csv"
        )
        logging.info("Ratios saved")


if __name__ == "__main__":
    args = set_args()
    main(args)
    # (1) Generate Synthetic Bandit Data
