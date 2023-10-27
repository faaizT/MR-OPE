import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def concatenate_feedback_data(bandit_feedback1, bandit_feedback2):
    bandit_feedback = {}
    if bandit_feedback1["n_actions"] != bandit_feedback2["n_actions"]:
        raise ValueError("n_actions should be same in both feedback dicts")
    bandit_feedback["action_context"] = bandit_feedback1["action_context"]
    bandit_feedback["n_rounds"] = (
        bandit_feedback1["n_rounds"] + bandit_feedback2["n_rounds"]
    )
    bandit_feedback["n_actions"] = bandit_feedback1["n_actions"]
    for key in bandit_feedback1:
        if key == "action_context":
            continue
        if bandit_feedback1[key] is None:
            bandit_feedback[key] = None
        elif (
            type(bandit_feedback1[key]) == np.ndarray
            and bandit_feedback1[key].shape[1:] == bandit_feedback2[key].shape[1:]
        ):
            bandit_feedback[key] = np.concatenate(
                [bandit_feedback1[key], bandit_feedback2[key]], axis=0
            )
    return bandit_feedback


def save_feedback_data(bandit_feedback, dir):
    for key in bandit_feedback:
        if (
            type(bandit_feedback[key]) == np.ndarray
            and len(bandit_feedback[key].shape) <= 2
        ):
            pd.DataFrame(bandit_feedback[key]).to_csv(f"{dir}/{key}.csv", index=None)
        elif (
            type(bandit_feedback[key]) == np.ndarray
            and len(bandit_feedback[key].shape) == 3
        ):
            pd.DataFrame(np.squeeze(bandit_feedback[key], axis=-1)).to_csv(
                f"{dir}/{key}.csv", index=None
            )


def load_feedback_data(dir):
    feedback = {}
    for path, subdirs, files in tqdm(os.walk(dir)):
        for name in files:
            key = name.split(".")[0]
            df = pd.read_csv(os.path.join(path, name))
            feedback[key] = df.to_numpy()
    feedback["position"] = None
    feedback["n_actions"] = feedback["action_context"].shape[0]
    feedback["n_rounds"] = feedback["context"].shape[0]
    feedback["pi_b"] = np.expand_dims(feedback["pi_b"], -1)
    feedback["reward"] = np.squeeze(feedback["reward"], -1)
    feedback["action"] = np.squeeze(feedback["action"], -1)
    feedback["pscore"] = np.squeeze(feedback["pscore"], -1)
    return feedback
