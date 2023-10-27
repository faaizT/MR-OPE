from typing import Optional
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from obp.utils import sample_action_fast, check_array
from obp.dataset import MultiClassToBanditReduction

from obp_utils.model_wrapper import PolicyWrapper


class MultiClassToBanditReductionAdapted(MultiClassToBanditReduction):
    """This class inherits the MultiClassToBanditReduction class,
    and in addition to returning only bandit feedback for evaluation fold,
    it also returns the bandit feedback for training fold.
    This training fold is then used to estimate the weights in MR estimator
    """

    def obtain_batch_bandit_feedback(
        self,
        beta_behav=1,
        random_state: Optional[int] = None,
        use_raw_behaviour_policy=False,
    ):
        """Obtain batch logged bandit data for evluation and training folds, an evaluation policy, and its ground-truth policy value.

        Note
        -------
        Please call `self.split_train_eval()` before calling this method.

        Parameters
        -----------
        random_state: int, default=None
            Controls the random seed in sampling actions.

        use_raw_behaviour_policy: bool, default=False
            if True, uses the classifier softmax scores as behaviour policy

        Returns
        ---------
        bandit_feedback_ev: BanditFeedback, bandit_feedback_tr: BanditFeedback
            bandit_feedback_ev (bandit_feedback_tr) is logged bandit data generated from a multi-class classification dataset for evaluation (training) fold.

        """
        random_ = check_random_state(random_state)
        # train a base ML classifier
        base_clf_b = self.base_classifier_b
        base_clf_b.fit(X=self.X_tr, y=self.y_tr)
        preds = base_clf_b.predict(self.X_ev).astype(int)
        preds_tr = base_clf_b.predict(self.X_tr).astype(int)
        # construct a behavior policy
        if use_raw_behaviour_policy:
            beh_policy = PolicyWrapper(classifier=base_clf_b, beta=beta_behav)
            pi_b = beh_policy.predict_proba(self.X_ev)
            pi_b_tr = beh_policy.predict_proba(self.X_tr)
        else:
            pi_b = np.zeros((self.n_rounds_ev, self.n_actions))
            pi_b[:, :] = (1.0 - self.alpha_b) / self.n_actions
            pi_b[np.arange(self.n_rounds_ev), preds] = (
                self.alpha_b + (1.0 - self.alpha_b) / self.n_actions
            )
            pi_b_tr = np.zeros((self.X_tr.shape[0], self.n_actions))
            pi_b_tr[:, :] = (1.0 - self.alpha_b) / self.n_actions
            pi_b_tr[np.arange(self.X_tr.shape[0]), preds_tr] = (
                self.alpha_b + (1.0 - self.alpha_b) / self.n_actions
            )
        if self.n_deficient_actions > 0:
            deficient_actions = np.argsort(
                random_.gumbel(size=(self.n_rounds_ev, self.n_actions)), axis=1
            )[:, ::-1][:, : self.n_deficient_actions]
            deficient_actions_idx = (
                np.tile(np.arange(self.n_rounds_ev), (self.n_deficient_actions, 1)).T,
                deficient_actions,
            )
            pi_b[deficient_actions_idx] = 0.0  # create some deficient actions
            pi_b /= pi_b.sum(1)[
                :, np.newaxis
            ]  # re-normalize the probability distribution

            deficient_actions_tr = np.argsort(
                random_.gumbel(size=(self.X_tr.shape[0], self.n_actions)), axis=1
            )[:, ::-1][:, : self.n_deficient_actions]
            deficient_actions_idx_tr = (
                np.tile(
                    np.arange(self.X_tr.shape[0]), (self.deficient_actions_tr, 1)
                ).T,
                deficient_actions_tr,
            )
            pi_b_tr[deficient_actions_idx_tr] = 0.0  # create some deficient actions
            pi_b_tr /= pi_b_tr.sum(1)[
                :, np.newaxis
            ]  # re-normalize the probability distribution
        # sample actions and factual rewards
        actions = sample_action_fast(pi_b, random_state=random_state)
        rewards = self.y_full_ev[np.arange(self.n_rounds_ev), actions]
        actions_tr = sample_action_fast(pi_b_tr, random_state=random_state)
        y_full_tr = np.eye(self.n_actions)[self.y_tr]
        rewards_tr = y_full_tr[np.arange(self.X_tr.shape[0]), actions_tr]

        return dict(
            n_actions=self.n_actions,
            n_rounds=self.n_rounds_ev,
            context=self.X_ev,
            action=actions,
            reward=rewards,
            position=None,  # position effect is not considered in classification data
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(self.n_rounds_ev), actions],
        ), dict(
            n_actions=self.n_actions,
            n_rounds=self.X_tr.shape[0],
            context=self.X_tr,
            action=actions_tr,
            reward=rewards_tr,
            position=None,  # position effect is not considered in classification data
            pi_b=pi_b_tr[:, :, np.newaxis],
            pscore=pi_b_tr[np.arange(self.X_tr.shape[0]), actions_tr],
        )

    def obtain_action_dist_by_eval_policy(
        self, base_classifier_e: Optional[ClassifierMixin] = None, alpha_e: float = 1.0
    ):
        """Obtain action choice probabilities by an evaluation policy for both training and evaluation data.

        Parameters
        -----------
        base_classifier_e: ClassifierMixin, default=None
            Machine learning classifier used to construct a behavior policy.

        alpha_e: float, default=1.0
            Ratio of a uniform random policy when constructing an **evaluation** policy.
            Must be in the [0, 1] interval (evaluation policy can be deterministic).

        Returns
        ---------
        action_dist_by_eval_policy: array-like, shape (n_rounds_ev, n_actions, 1), action_dist_by_eval_policy_tr: array-like, shape (n_rounds_tr, n_actions, 1)
            `action_dist_by_eval_policy` is the action choice probabilities of the evaluation policy for evaluation fold.
            `action_dist_by_eval_policy_tr` is the action choice probabilities of the evaluation policy for training fold.
            where `n_rounds_ev` (`n_rounds_tr`) is the number of samples in the evaluation (training) set given the current train-eval split.
            `n_actions` is the number of actions.

        """
        check_scalar(alpha_e, "alpha_e", float, min_val=0.0, max_val=1.0)
        # train a base ML classifier
        if base_classifier_e is None:
            base_clf_e = self.base_classifier_b
        else:
            assert is_classifier(
                base_classifier_e
            ), "`base_classifier_e` must be a classifier"
            base_clf_e = base_classifier_e
        preds = base_clf_e.predict(self.X_ev).astype(int)
        preds_tr = base_clf_e.predict(self.X_tr).astype(int)
        preds_full = base_clf_e.predict(self.X).astype(int)
        # construct an evaluation policy
        pi_e = np.zeros((self.n_rounds_ev, self.n_actions))
        pi_e[:, :] = (1.0 - alpha_e) / self.n_actions
        pi_e[np.arange(self.n_rounds_ev), preds] = (
            alpha_e + (1.0 - alpha_e) / self.n_actions
        )
        pi_e_tr = np.zeros((self.X_tr.shape[0], self.n_actions))
        pi_e_tr[:, :] = (1.0 - alpha_e) / (self.n_actions - 1)
        pi_e_tr[np.arange(self.X_tr.shape[0]), preds_tr] = alpha_e
        pi_e_full = np.zeros((self.X.shape[0], self.n_actions))
        pi_e_full[:, :] = (1.0 - alpha_e) / (self.n_actions - 1)
        pi_e_full[np.arange(self.X.shape[0]), preds_full] = alpha_e
        return (
            pi_e[:, :, np.newaxis],
            pi_e_tr[:, :, np.newaxis],
            pi_e_full[:, :, np.newaxis],
        )

    def obtain_action_dist_by_eval_policy_using_beta(
        self,
        base_classifier_e: Optional[ClassifierMixin] = None,
        beta_target: float = 1.0,
    ):
        """Obtain action choice probabilities by an evaluation policy for both training and evaluation data.

        Parameters
        -----------
        base_classifier_e: ClassifierMixin, default=None
            Machine learning classifier used to construct a behavior policy.

        beta_target: float, default=1.0
            Ratio of a uniform random policy when constructing an **evaluation** policy.

        Returns
        ---------
        action_dist_by_eval_policy: array-like, shape (n_rounds_ev, n_actions, 1), action_dist_by_eval_policy_tr: array-like, shape (n_rounds_tr, n_actions, 1)
            `action_dist_by_eval_policy` is the action choice probabilities of the evaluation policy for evaluation fold.
            `action_dist_by_eval_policy_tr` is the action choice probabilities of the evaluation policy for training fold.
            where `n_rounds_ev` (`n_rounds_tr`) is the number of samples in the evaluation (training) set given the current train-eval split.
            `n_actions` is the number of actions.

        """
        # train a base ML classifier
        if base_classifier_e is None:
            base_clf_e = self.base_classifier_b
        else:
            assert is_classifier(
                base_classifier_e
            ), "`base_classifier_e` must be a classifier"
            base_clf_e = base_classifier_e
        # construct an evaluation policy
        target_policy = PolicyWrapper(base_clf_e, beta=beta_target)
        pi_e = target_policy.predict_proba(self.X_ev)
        pi_e_tr = target_policy.predict_proba(self.X_tr)
        pi_e_full = target_policy.predict_proba(self.X)
        return (
            pi_e[:, :, np.newaxis],
            pi_e_tr[:, :, np.newaxis],
            pi_e_full[:, :, np.newaxis],
        )

    def calc_ground_truth_policy_value(self, action_dist: np.ndarray) -> float:
        """Calculate the ground-truth policy value of the given action distribution.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds_ev, n_actions, 1)
            Action distribution or action choice probabilities of a policy whose ground-truth is to be caliculated here.
            where `n_rounds_ev` is the number of samples in the evaluation set given the current train-eval split.
            `n_actions` is the number of actions.

        Returns
        ---------
        ground_truth_policy_value: float
            policy value of given action distribution (mostly evaluation policy).

        """
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if action_dist.shape[0] == self.n_rounds_ev:
            return action_dist[np.arange(self.n_rounds_ev), self.y_ev].mean()
        elif action_dist.shape[0] == self.n_rounds:
            return action_dist[np.arange(self.n_rounds), self.y].mean()
        else:
            raise ValueError(
                "Expected `action_dist.shape[0] == self.n_rounds_ev` or `action_dist.shape[0] == self.n_rounds`, but found it False"
            )
