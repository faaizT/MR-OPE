from ast import Dict
from typing import Optional
import numpy as np
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import DoublyRobustWithShrinkageTuning as DRos
from obp.ope import InverseProbabilityWeighting as IPW
from obp.ope import MarginalizedInverseProbabilityWeighting as MIPS
from obp.ope import OffPolicyEvaluation
from obp.ope import SubGaussianDoublyRobustTuning as SGDR
from obp.ope import SwitchDoublyRobustTuning as SwitchDR
from obp_utils.estimator_mr import MarginalizedRatioWeighting as MR


def create_ope_object(
    bandit_feedback_test: Dict,
    use_train_behav: bool,
    pi_b_scores: np.ndarray,
    pi_e_scores: np.ndarray,
    reward: np.ndarray,
    reward_type: str,
    estimation_method="default",
):
    lambdas = [10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, np.inf]
    lambdas_sg = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback_test,
        ope_estimators=[
            IPW(use_estimated_pscore=use_train_behav),
            DM(),
            DR(use_estimated_pscore=use_train_behav),
            SwitchDR(
                lambdas=lambdas,
                tuning_method="slope",
                estimator_name="SwitchDR",
                use_estimated_pscore=use_train_behav,
            ),
            DRos(
                lambdas=lambdas,
                tuning_method="slope",
                estimator_name="DRos",
                use_estimated_pscore=use_train_behav,
            ),
            MR(
                reward=reward,
                pi_b=pi_b_scores,
                pi_e=pi_e_scores,
                reward_type=reward_type,
                estimation_method=estimation_method,
                use_estimated_pscore=use_train_behav,
            ),
        ],
    )
    return ope
