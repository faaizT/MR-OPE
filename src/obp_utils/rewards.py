import numpy as np
from typing import Optional
from scipy.special import softmax


def reward_context_times_action(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function q(x, a) = x.sum()*a for continuous rewards.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return context.sum(1).reshape(-1, 1).repeat(action_context.shape[0], 1) * np.arange(
        action_context.shape[0]
    ).reshape(1, -1)


def reward_indep_action(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function independent of action for continuous rewards.

    Note
    ------
    reward and actions are indepdent in this example

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return (np.exp(context)).sum(1).reshape(-1, 1).repeat(action_context.shape[0], 1)


def reward_context_norm_plus_action(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function independent of action for continuous rewards.

    Note
    ------
    q(x, a) = ||x|| + a

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return (
        (np.linalg.norm(context, axis=-1))
        .reshape(-1, 1)
        .repeat(action_context.shape[0], 1)
    ) + np.arange(action_context.shape[0]).reshape(1, -1)


def reward_context_sin_norm(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function independent of action for continuous rewards.

    Note
    ------
    q(x, a) = sin(2 * ||x||)

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return (
        (np.sin(2 * np.linalg.norm(context, axis=-1)))
        .reshape(-1, 1)
        .repeat(action_context.shape[0], 1)
    )


def reward_sin_ax(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function independent of action for continuous rewards.

    Note
    ------
    q(x, a) = sin(a * ||x||)

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return np.sin(
        np.linalg.norm(context, axis=-1).reshape(-1, 1)
        * np.arange(action_context.shape[0]).reshape(1, -1)
    )


def reward_sin_ax_normalised(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function independent of action for continuous rewards.

    Note
    ------
    q(x, a) = sin(2pi * a/n_actions * ||x||/sqrt(d))

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return np.sin(
        2
        * np.pi
        * np.linalg.norm(context, axis=-1).reshape(-1, 1)
        * np.arange(action_context.shape[0]).reshape(1, -1)
        / (action_context.shape[0])
        / np.sqrt(context.shape[1])
    )


def reward_exp_ax_normalised(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function independent of action for continuous rewards.

    Note
    ------
    q(x, a) = exp(-a/n_actions * ||x||/sqrt(d))

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return np.exp(
        -np.linalg.norm(context, axis=-1).reshape(-1, 1)
        * np.arange(action_context.shape[0]).reshape(1, -1)
        / (action_context.shape[0])
        / np.sqrt(context.shape[1])
    )


def reward_ax_first_5_components(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function independent of action for continuous rewards.

    Note
    ------
    q(x, a) = a * ||x_{0:4}||

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return (
        np.linalg.norm(context[:, :5], axis=-1).reshape(-1, 1)
        * np.arange(action_context.shape[0]).reshape(1, -1)
        / (action_context.shape[0] - 1)
    )


def reward_sine_ax_first_5_components(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function independent of action for continuous rewards.

    Note
    ------
    q(x, a) = sin(a * ||x_{0:4}||)

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return np.sin(
        np.linalg.norm(context[:, :5], axis=-1).reshape(-1, 1)
        * np.arange(action_context.shape[0]).reshape(1, -1)
        / (action_context.shape[0] - 1)
    )


def reward_sin_ax_plus_one(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function independent of action for continuous rewards.

    Note
    ------
    q(x, a) = (sin(a * ||x||) + 1)/2

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return (
        np.sin(
            np.linalg.norm(context, axis=-1).reshape(-1, 1)
            * np.arange(action_context.shape[0]).reshape(1, -1)
        )
        + 1
    ) / 2


def reward_binary(
    context: np.ndarray,
    action_context: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Reward function for binary rewards with q(x, a) = softmax(exp(a)).

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors characterizing each data (such as user information).

    action_context: array-like, shape (n_actions, dim_action_context)
        Vector representation of actions.

    random_state: int, default=None
        Controls the random seed in sampling dataset.

    Returns
    ---------
    expected_rewards: array-like, shape (n_rounds, n_actions)
        Expected reward given context (:math:`x`) and action (:math:`a`),
        i.e., :math:`q(x,a):=\\mathbb{E}[r|x,a]`.

    """
    return softmax(
        np.exp(np.arange(action_context.shape[0]))
        .reshape(1, -1)
        .repeat(context.shape[0], axis=0),
        axis=-1,
    )
