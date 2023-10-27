import argparse
from logging import getLogger
import logging
from pathlib import Path
from time import time
import warnings

import numpy as np
from obp.dataset import linear_reward_function
from obp.dataset import SyntheticBanditDatasetWithActionEmbeds
from obp.ope import RegressionModel
from mips_experiment_utils.ope import run_ope
import pandas as pd
from pandas import DataFrame
from mips_experiment_utils.policy import gen_eps_greedy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning

from utils.helper_functions import create_tag, str2bool


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = getLogger(__name__)


def set_args():
    parser = argparse.ArgumentParser(description="Parser for the MIPS OPE experiments")
    parser.add_argument("--n_action", type=int, default=10, help="Number of actions")
    parser.add_argument(
        "--results_dir", default=".", type=str, help="Folder to save the results"
    )
    parser.add_argument(
        "--random_state",
        default=12345,
        type=int,
        help="Random state for the experiments",
    )
    parser.add_argument("--seed", type=int, default=5, help="Seed for the run")
    parser.add_argument(
        "--dim_context", type=int, default=1000, help="Context dimension"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=-1,
        help="Beta parameter for the behaviour policy",
    )
    parser.add_argument(
        "--latent_param_mat_dim",
        type=int,
        default=5,
        help="Latent parameter dimension",
    )
    parser.add_argument("--n_cat_dim", type=int, default=3, help="N_cat dimension")
    parser.add_argument(
        "--n_unobserved_cat_dim",
        type=int,
        default=0,
        help="Unobserved n_cat dimension",
    )
    parser.add_argument(
        "--n_def_actions",
        type=int,
        default=0,
        help="Number of deficient actions",
    )
    parser.add_argument(
        "--reward_std",
        type=float,
        default=2.5,
        help="Standard deviation of p(Y|X, A)",
    )
    parser.add_argument(
        "--n_cat_per_dim",
        type=int,
        default=10,
        help="Number of cat per dimension",
    )
    parser.add_argument(
        "--n_test_data",
        type=int,
        default=500,
        help="Number of training points for OPE",
    )
    parser.add_argument(
        "--n_val_data",
        type=int,
        default=4500,
        help="Number of validation datapoints",
    )
    parser.add_argument(
        "--is_optimal",
        type=str2bool,
        default="True",
        help="Is optimal or not",
    )
    parser.add_argument(
        "--embed_selection",
        type=str2bool,
        default="False",
        help="Embed selection",
    )
    parser.add_argument(
        "--use_train_behav",
        type=str2bool,
        default="True",
        help="Train behaviour policy",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.2,
        help="Eps parameter for target policy",
    )
    args = parser.parse_args()
    return args


def main(args) -> None:
    print(args)
    logging.info("starting the job")
    use_train_behav = args.use_train_behav
    logger.info(f"The current working directory is {Path().cwd()}")
    start_time = time()

    # results path
    Path(args.results_dir).mkdir(exist_ok=True, parents=True)
    random_state = args.random_state

    # set seed
    np.random.seed(args.seed)

    elapsed_prev = 0.0
    estimated_policy_value_list = []
    ## define a dataset class
    logging.info("creating the data")
    dataset = SyntheticBanditDatasetWithActionEmbeds(
        n_actions=args.n_action,
        dim_context=args.dim_context,
        beta=args.beta,
        reward_type="continuous",
        n_cat_per_dim=args.n_cat_per_dim,
        latent_param_mat_dim=args.latent_param_mat_dim,
        n_cat_dim=args.n_cat_dim,
        n_unobserved_cat_dim=args.n_unobserved_cat_dim,
        n_deficient_actions=int(args.n_action * args.n_def_actions),
        reward_function=linear_reward_function,
        reward_std=args.reward_std,
        random_state=args.seed,
    )
    logging.info("created the data")
    ### test bandit data is used to approximate the ground-truth policy value
    test_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=args.n_test_data)
    logging.info("got the test data")
    action_dist_test = gen_eps_greedy(
        expected_reward=test_bandit_data["expected_reward"],
        is_optimal=args.is_optimal,
        eps=args.eps,
    )
    policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=test_bandit_data["expected_reward"],
        action_dist=action_dist_test,
    )
    logging.info("got the gt policy value")

    ## generate validation data
    val_bandit_data = dataset.obtain_batch_bandit_feedback(n_rounds=args.n_val_data)
    logging.info("got the val data")

    ## make decisions on validation data
    action_dist_val = gen_eps_greedy(
        expected_reward=val_bandit_data["expected_reward"],
        is_optimal=args.is_optimal,
        eps=args.eps,
    )

    ## OPE using validation data
    reg_model = RegressionModel(
        n_actions=dataset.n_actions,
        action_context=val_bandit_data["action_context"],
        base_model=RandomForestRegressor(
            n_estimators=10,
            max_samples=0.8,
            random_state=random_state + args.seed,
            max_depth=2,
        ),
    )
    estimated_rewards = reg_model.fit_predict(
        context=val_bandit_data["context"],  # context; x
        action=val_bandit_data["action"],  # action; a
        reward=val_bandit_data["reward"],  # reward; r
        n_folds=2,
        random_state=random_state + args.seed,
    )
    logging.info("fitting the regression model")

    if use_train_behav:
        logging.info("Fitting behaviour policy")
        # Use half the training data to train the behaviour policy
        behav_policy = RandomForestClassifier(
            n_estimators=25, max_samples=0.8, max_depth=8
        )
        behav_policy.fit(
            X=test_bandit_data["context"][: args.n_test_data // 2],
            y=test_bandit_data["action"][: args.n_test_data // 2],
        )
        pi_b_all_actions_training = behav_policy.predict_proba(
            X=test_bandit_data["context"]
        )
        pi_b_all_actions_testing = behav_policy.predict_proba(
            X=val_bandit_data["context"]
        )
        pi_b_scores_train = np.clip(
            pi_b_all_actions_training[
                np.arange(pi_b_all_actions_training.shape[0]),
                test_bandit_data["action"],
            ],
            0.0001,
            0.9999,
        )
    else:
        pi_b_all_actions_testing = val_bandit_data["pi_b"]
        pi_b_scores_train = test_bandit_data["pscore"]

    logging.info("running the ope")
    estimated_policy_values = run_ope(
        val_bandit_data=val_bandit_data,
        test_bandit_data=test_bandit_data,
        action_dist_val=action_dist_val,
        pi_e_test=action_dist_test[
            np.arange(args.n_test_data), test_bandit_data["action"], 0
        ],
        estimated_rewards=estimated_rewards,
        embed_selection=args.embed_selection,
        use_train_behav=use_train_behav,
        pi_b_test=pi_b_scores_train,
        pi_b=np.expand_dims(pi_b_all_actions_testing, -1),
    )
    estimated_policy_value_list.append(estimated_policy_values)

    ## summarize results
    result_df = (
        DataFrame(DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "value"})
    )
    logging.info("got the ope results")
    result_df["n_action"] = args.n_action
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = result_df.value - policy_value
    result_df["ground_truth_value"] = policy_value

    elapsed = np.round((time() - start_time) / 60, 2)
    diff = np.round(elapsed - elapsed_prev, 2)
    logger.info(f"n_action={args.n_action}: {elapsed}min (diff {diff}min)")
    elapsed_prev = elapsed

    # save results
    ignored_cols = [
        "results_dir",
        "latent_param_mat_dim",
        "n_unobserved_cat_dim",
        "is_optimal",
    ]
    tag = create_tag(args, ignore_cols=ignored_cols)
    args_df = pd.DataFrame(vars(args), index=result_df.index)
    pd.concat([result_df, args_df], axis=1).reset_index(drop=False).to_csv(
        f"{args.results_dir}/results_{tag}.csv", index=False
    )
    logging.info(pd.concat([result_df, args_df], axis=1).reset_index(drop=False))
    logging.info("Results saved")


if __name__ == "__main__":
    args = set_args()
    main(args)
