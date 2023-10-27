# implementing an experiment to evaluate the accuracy of OPE using classification data
import argparse
from pathlib import Path
from sklearn import clone
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from keras.datasets import mnist
import numpy as np
import pandas as pd

# import open bandit pipeline (obp)
from obp.dataset import MultiClassToBanditReduction
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting as IPW,
    DirectMethod as DM,
    DoublyRobust as DR,
)
from obp_utils.data_loader_funcs import load_data
from obp_utils.estimator_mr import MarginalizedRatioWeighting as MR
from obp_utils.multiclass import MultiClassToBanditReductionAdapted
import logging

from obp_utils.ope import create_ope_object
from utils.helper_functions import create_tag, str2bool

"""
This is a tutorial implementation of OPE using the OBP library (https://github.com/st-tech/zr-obp).
Here, we use the library to run OPE experiments for IPW, DM and DR estimators using synthetic data.
"""


def set_args():
    parser = argparse.ArgumentParser(
        description="Parser for the OPE Multiclass experiments"
    )
    parser.add_argument(
        "--results_dir", default=".", type=str, help="Folder to save the results"
    )
    parser.add_argument(
        "--n_eval", type=int, default=1000, help="Number of evaluation points for OPE"
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=1000,
        help="Number of training points for weights",
    )
    parser.add_argument("--seed", type=int, default=2921, help="Seed for the run")
    parser.add_argument(
        "--beta_behav",
        type=float,
        default=1.0,
        help="Beta parameter for the behavioural policy",
    )
    parser.add_argument(
        "--beta_target",
        type=float,
        default=1.0,
        help="Beta parameter for the target policy",
    )
    parser.add_argument(
        "--alpha_b",
        type=float,
        default=0.0,
        help="Alpha parameter for the behaviour policy",
    )
    parser.add_argument(
        "--alpha_e",
        type=float,
        default=0.2,
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
        default="True",
        help="save_ratios",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="digits",
        help="The dataset to run experiment on.",
    )
    args = parser.parse_args()
    return args


def main(args):
    # (1) Generate MultiClass Bandit Data
    reward_type = "binary"
    data = args.data
    beta_behav = args.beta_behav
    beta_target = args.beta_target
    n_eval = args.n_eval
    alpha_e = args.alpha_e
    alpha_b = args.alpha_b
    seed = args.seed
    use_train_behav = args.use_train_behav
    n_train = args.n_train
    np.random.seed(seed)
    X, y = load_data(data)

    # convert the raw classification data into a logged bandit dataset
    # we construct a behavior policy using Logistic Regression and parameter `alpha_b`
    # given a pair of a feature vector and a label (x, c), create a pair of a context vector and reward (x, r)
    # where r = 1 if the output of the behavior policy is equal to c, and r = 0 otherwise
    dataset = MultiClassToBanditReductionAdapted(
        X=X,
        y=y,
        base_classifier_b=LogisticRegression(),
        dataset_name=data,
        alpha_b=alpha_b,
    )

    # split the original data into training and evaluation sets
    dataset.split_train_eval(eval_size=n_eval, random_state=seed)

    # get training and testing data
    testing_feedback, training_feedback = dataset.obtain_batch_bandit_feedback(
        beta_behav=beta_behav, use_raw_behaviour_policy=True, random_state=seed
    )
    # n_train = training_feedback["action"].shape[0]

    # obtain action choice probabilities of an evaluation policy
    # we construct an evaluation policy using Random Forest and parameter `alpha_e`

    """
    The following commented out code defines an alternative target policy class
    """
    eval_policy_classifier = LogisticRegression()
    eval_policy_classifier.fit(
        X=dataset.X_tr[: n_train // 2], y=dataset.y_tr[: n_train // 2]
    )
    (
        action_dist_test,
        action_dist_tr,
        action_dist_full,
    ) = dataset.obtain_action_dist_by_eval_policy(
        base_classifier_e=eval_policy_classifier, alpha_e=alpha_e
    )

    pi_e_scores = action_dist_tr[
        np.arange(action_dist_tr.shape[0]), training_feedback["action"], 0
    ]

    # estimate the expected rewards by using an ML model (Logistic Regression here)
    # the estimated rewards are used by model-dependent estimators such as DM and DR
    logging.info("Fitting the reward model")
    regression_model = RandomForest(
        n_estimators=10,
        max_samples=0.8,
        max_depth=2,
        random_state=args.seed,
    )
    regression_model.fit(training_feedback["context"][:n_train], dataset.y_tr[:n_train])

    estimated_rewards_by_reg_model = np.expand_dims(
        regression_model.predict_proba(testing_feedback["context"]), axis=-1
    )

    if use_train_behav:
        logging.info("Fitting behaviour policy")
        # Use half the training data to train the behaviour policy
        behav_policy = LogisticRegression()
        behav_policy.fit(
            X=training_feedback["context"][:n_train],
            y=training_feedback["action"][:n_train],
        )
        pi_b_all_actions_training = behav_policy.predict_proba(
            X=training_feedback["context"]
        )
        pi_b_all_actions_testing = behav_policy.predict_proba(
            X=testing_feedback["context"]
        )
        pi_b_scores_train = np.clip(
            pi_b_all_actions_training[
                np.arange(pi_b_all_actions_training.shape[0]),
                training_feedback["action"],
            ],
            0.0001,
            0.9999,
        )
        pi_b_scores_test = np.clip(
            pi_b_all_actions_testing[
                np.arange(pi_b_all_actions_testing.shape[0]),
                testing_feedback["action"],
            ],
            0.0001,
            0.9999,
        )
    else:
        pi_b_scores_test = testing_feedback["pscore"]
        pi_b_scores_train = training_feedback["pscore"]

    # Use the other half of training data to train the weights for Marginal ratios
    ope = create_ope_object(
        bandit_feedback_test=testing_feedback,
        use_train_behav=use_train_behav,
        pi_b_scores=pi_b_scores_train[:n_train],
        pi_e_scores=pi_e_scores[:n_train],
        reward=training_feedback["reward"][:n_train],
        reward_type=reward_type,
    )

    # calculate ground truth policy value
    ground_truth = dataset.calc_ground_truth_policy_value(
        action_dist=action_dist_full,
    )
    logging.info(f"Ground truth policy value: {ground_truth}")

    # estimate the policy value of IPWLearner with Logistic Regression
    estimated_policy_value_a, estimated_interval_a = ope.summarize_off_policy_estimates(
        estimated_pscore=pi_b_scores_test,
        action_dist=action_dist_test,
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

    relative_estimation_errors = ope.summarize_estimators_comparison(
        estimated_pscore=pi_b_scores_test,
        ground_truth_policy_value=ground_truth,
        action_dist=action_dist_test,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        metric="relative-ee",
    )
    relative_estimation_errors["ground_truth_value"] = ground_truth
    logging.info("Relative estimation errors for policy value:")
    logging.info("\n" + str(relative_estimation_errors))

    squared_errors = ope.summarize_estimators_comparison(
        estimated_pscore=pi_b_scores_test,
        ground_truth_policy_value=ground_truth,
        action_dist=action_dist_test,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        metric="se",
    )
    logging.info("Squared errors for policy value:")
    logging.info("\n" + str(squared_errors))

    results_df = pd.concat(
        [
            relative_estimation_errors,
            squared_errors,
            estimated_policy_value_a,
            estimated_interval_a,
        ],
        axis=1,
    )
    logging.info("Results:")
    logging.info(results_df)
    tag = create_tag(args, ignore_cols=["results_dir", "save_ratios"])
    args_df = pd.DataFrame(vars(args), index=results_df.index)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    pd.concat([results_df, args_df], axis=1).reset_index(drop=False).to_csv(
        f"{args.results_dir}/results_multiclass_{tag}.csv", index=False
    )
    logging.info("Results saved")
    # Save the ratio values for policy ratios as well as our method
    if args.save_ratios:
        ope.ope_estimators[-1].save_ratio_values(
            file_path=f"{args.results_dir}/ratios_{tag}.csv"
        )
        logging.info("Ratios saved")


if __name__ == "__main__":
    args = set_args()
    main(args)
