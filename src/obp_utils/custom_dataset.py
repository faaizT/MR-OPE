import logging
from pathlib import Path
from typing import Optional, Union
import dowhy
import numpy as np
from obp.dataset import OpenBanditDataset
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
import dowhy.datasets
from tqdm import tqdm


class CustomBanditDataset(OpenBanditDataset):
    def __init__(self):
        self.load_raw_data()

    def __post_init__(self):
        return

    def load_raw_data(self):
        raise NotImplementedError("To be implemented for child classes")

    def obtain_batch_bandit_feedback(
        self,
        test_size: int = 100,
    ):
        """Obtain batch logged bandit data.

        Parameters
        -----------
        test_size: float, default=0.3
            Proportion of the dataset included in the test split.
            If float, should be between 0.0 and 1.0.
            This argument matters only when `is_timeseries_split=True` (the out-sample case).

        is_timeseries_split: bool, default=False
            If true, split the original logged bandit data into train and test sets based on time series.

        Returns
        --------
        bandit_feedback: BanditFeedback
            A dictionary containing batch logged bandit data collected by the behavior policy.
            The keys of the dictionary are as follows.
            - n_rounds: number of rounds, data size of the logged bandit data
            - n_actions: number of actions (:math:`|\mathcal{A}|`)
            - action: action variables sampled by the behavior policy
            - position: positions where actions are recommended, there are three positions in the ZOZOTOWN rec interface
            - reward: binary reward variables, click indicators
            - pscore: action choice probabilities by the behavior policy, propensity scores
            - context: context vectors such as user-related features and user-item affinity scores
            - action_context: item-related context vectors

        """

        n_rounds_train = self.n_rounds - test_size
        bandit_feedback_train = dict(
            n_rounds=n_rounds_train,
            n_actions=self.n_actions,
            action=self.action[:n_rounds_train],
            reward=self.reward[:n_rounds_train],
            context=self.context[:n_rounds_train],
            position=None,
        )
        bandit_feedback_test = dict(
            n_rounds=(self.n_rounds - n_rounds_train),
            n_actions=self.n_actions,
            action=self.action[n_rounds_train:],
            reward=self.reward[n_rounds_train:],
            context=self.context[n_rounds_train:],
            position=None,
        )
        self.data_test = self.data.iloc[n_rounds_train:, :]

        return bandit_feedback_train, bandit_feedback_test

    def sample_bootstrap_bandit_feedback(
        self,
        sample_size: Optional[int] = None,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
        random_state: Optional[int] = None,
    ):
        """Obtain bootstrap logged bandit feedback.

        Parameters
        -----------
        sample_size: int, default=None
            Number of data sampled by bootstrap.
            If None, the original data size (n_rounds) is used as `sample_size`.
            The value must be smaller than the original data size.

        test_size: float, default=0.3
            Proportion of the dataset included in the test split.
            If float, should be between 0.0 and 1.0.
            This argument matters only when `is_timeseries_split=True` (the out-sample case).

        is_timeseries_split: bool, default=False
            If true, split the original logged bandit data into train and test sets based on time series.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        --------
        bandit_feedback: BanditFeedback
            A dictionary containing logged bandit data collected by the behavior policy.
            The keys of the dictionary are as follows.
            - n_rounds: number of rounds, data size of the logged bandit data
            - n_actions: number of actions (:math:`|\mathcal{A}|`)
            - action: action variables sampled by the behavior policy
            - position: positions where actions are recommended, there are three positions in the ZOZOTOWN rec interface
            - reward: binary reward variables, click indicators
            - pscore: action choice probabilities by the behavior policy, propensity scores
            - context: context vectors such as user-related features and user-item affinity scores
            - action_context: item-related context vectors

        """
        bandit_feedback = self.obtain_batch_bandit_feedback(test_size=test_size)[0]
        n_rounds = bandit_feedback["n_rounds"]
        if sample_size is None:
            sample_size = bandit_feedback["n_rounds"]
        else:
            check_scalar(
                sample_size,
                name="sample_size",
                target_type=(int),
                min_val=0,
                max_val=n_rounds,
            )
        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(
            np.arange(n_rounds), size=sample_size, replace=True
        )
        for key_ in ["action", "reward", "context", "position"]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]
        bandit_feedback["n_rounds"] = sample_size
        return bandit_feedback


class IHDPDataset(CustomBanditDataset):
    dataset_name = "ihdp"

    def load_raw_data(self):
        """Load raw IHDP bandit dataset."""

        data = pd.read_csv(
            "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
            header=None,
        )
        col = [
            "treatment",
            "y_factual",
            "y_cfactual",
            "mu0",
            "mu1",
        ]
        for i in range(1, 26):
            col.append("x" + str(i))
        context_cols = col[3:]
        data.columns = col
        data = data.astype({"treatment": "bool"}, copy=False).sample(frac=1)

        data_1 = data[data["treatment"] == 1]
        data_0 = data[data["treatment"] == 0]
        self.ground_truth_ate = np.mean(
            (data["treatment"] == 1) * (data["y_factual"] - data["y_cfactual"])
            + (data["treatment"] == 0) * (data["y_cfactual"] - data["y_factual"])
        )
        include = (data["treatment"] == 0) & (
            np.random.binomial(n=1, p=0.1, size=(len(data)))
        ) | (data["treatment"] == 1)
        self.data = data[include]
        self.reward = self.data["y_factual"].values
        self.action = self.data["treatment"].apply(int).values
        self.position = None
        self.context = self.data.loc[:, context_cols].values
        self.outcome = "y_factual"
        self.treatment = "treatment"
        self.context_cols = context_cols
        self.reward_type = "continuous"


class LalondeDataset(CustomBanditDataset):
    dataset_name = "lalonde"

    def load_raw_data(self):
        """Load raw lalonde bandit dataset."""
        lalonde = dowhy.datasets.lalonde_dataset().sample(frac=1)
        context_cols = ["nodegr", "black", "hisp", "age", "educ", "married"]
        self.context = lalonde.loc[:, context_cols].values
        self.action = lalonde["treat"].astype(int).values
        self.reward = lalonde["re78"].values
        self.position = None
        self.ground_truth_ate = (
            lalonde.loc[lalonde["treat"] == 1, "re78"].mean()
            - lalonde.loc[lalonde["treat"] == 0, "re78"].mean()
        )
        self.data = lalonde
        self.outcome = "re78"
        self.treatment = "treat"
        self.context_cols = context_cols
        self.reward_type = "continuous"


class CriteoDataset(CustomBanditDataset):
    dataset_name = "criteo"

    def __init__(self, balance_param=0.5):
        self.balance_param = balance_param
        super().__init__()

    def load_raw_data(self):
        """Load raw lalonde bandit dataset."""
        criteo_full = pd.read_csv(
            "/data/ziz/not-backed-up/taufiq/criteo-uplift-v2.1.csv"
        ).sample(frac=1)
        self.ground_truth_ate = (
            criteo_full.loc[criteo_full["treatment"] == 1, "visit"].mean()
            - criteo_full.loc[criteo_full["treatment"] == 0, "visit"].mean()
        )
        criteo_t1 = criteo_full.loc[criteo_full["treatment"] == 1, :].reset_index(
            drop=True
        )
        criteo_t0 = criteo_full.loc[criteo_full["treatment"] == 0, :].reset_index(
            drop=True
        )
        n1 = int(10000 * self.balance_param)
        n0 = 10000 - n1
        criteo_t1, criteo_t0 = criteo_t1.iloc[:n1], criteo_t0.iloc[:n0]
        criteo = pd.concat([criteo_t1, criteo_t0], ignore_index=True).sample(frac=1)
        context_cols = [f"f{i}" for i in range(12)]
        self.context = criteo.loc[:, context_cols].values
        self.action = criteo["treatment"].astype(int).values
        self.reward = criteo["visit"].values
        self.position = None
        self.data = criteo
        self.outcome = "visit"
        self.treatment = "treatment"
        self.context_cols = context_cols
        self.reward_type = "binary"


class TwinsDataset(CustomBanditDataset):
    dataset_name = "twins"
    cols = [
        "pldel",
        "birattnd",
        "brstate",
        "stoccfipb",
        "mager8",
        "ormoth",
        "mrace",
        "meduc6",
        "dmar",
        "mplbir",
        "mpre5",
        "adequacy",
        "orfath",
        "frace",
        "birmon",
        "gestat10",
        "csex",
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
        "tobacco",
        "alcohol",
        "cigar6",
        "drink5",
        "crace",
        "data_year",
        "nprevistq",
        "dfageq",
        "feduc6",
        "infant_id",
        "dlivord_min",
        "dtotord_min",
        "bord",
        "brstate_reg",
        "stoccfipb_reg",
        "mplbir_reg",
        "wt",
        "treatment",
        "outcome",
    ]

    def __init__(self, dataset_file, balance_param=0.5):
        self.dataset_file = dataset_file
        self.balance_param = balance_param
        if Path(dataset_file).is_file():
            self.load_raw_data()
        else:
            self.pre_process()
            self.load_raw_data()

    def pre_process(self):
        """Load raw Twins bandit dataset."""
        logging.info("preprocessing twin file")
        x = pd.read_csv(
            "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_X_3years_samesex.csv"
        )

        # The outcome data contains mortality of the lighter and heavier twin
        y = pd.read_csv(
            "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv"
        )

        # The treatment data contains weight in grams of both the twins
        t = pd.read_csv(
            "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_samesex.csv"
        )

        # _0 denotes features specific to the lighter twin and _1 denotes features specific to the heavier twin
        lighter_columns = [
            "pldel",
            "birattnd",
            "brstate",
            "stoccfipb",
            "mager8",
            "ormoth",
            "mrace",
            "meduc6",
            "dmar",
            "mplbir",
            "mpre5",
            "adequacy",
            "orfath",
            "frace",
            "birmon",
            "gestat10",
            "csex",
            "anemia",
            "cardiac",
            "lung",
            "diabetes",
            "herpes",
            "hydra",
            "hemo",
            "chyper",
            "phyper",
            "eclamp",
            "incervix",
            "pre4000",
            "preterm",
            "renal",
            "rh",
            "uterine",
            "othermr",
            "tobacco",
            "alcohol",
            "cigar6",
            "drink5",
            "crace",
            "data_year",
            "nprevistq",
            "dfageq",
            "feduc6",
            "infant_id_0",
            "dlivord_min",
            "dtotord_min",
            "bord_0",
            "brstate_reg",
            "stoccfipb_reg",
            "mplbir_reg",
        ]
        heavier_columns = [
            "pldel",
            "birattnd",
            "brstate",
            "stoccfipb",
            "mager8",
            "ormoth",
            "mrace",
            "meduc6",
            "dmar",
            "mplbir",
            "mpre5",
            "adequacy",
            "orfath",
            "frace",
            "birmon",
            "gestat10",
            "csex",
            "anemia",
            "cardiac",
            "lung",
            "diabetes",
            "herpes",
            "hydra",
            "hemo",
            "chyper",
            "phyper",
            "eclamp",
            "incervix",
            "pre4000",
            "preterm",
            "renal",
            "rh",
            "uterine",
            "othermr",
            "tobacco",
            "alcohol",
            "cigar6",
            "drink5",
            "crace",
            "data_year",
            "nprevistq",
            "dfageq",
            "feduc6",
            "infant_id_1",
            "dlivord_min",
            "dtotord_min",
            "bord_1",
            "brstate_reg",
            "stoccfipb_reg",
            "mplbir_reg",
        ]
        # Since data has pair property,processing the data to get separate row for each twin so that each child can be treated as an instance
        data = []

        for i in tqdm(range(len(t.values))):

            # select only if both <=2kg
            if t.iloc[i].values[1] >= 2000 or t.iloc[i].values[2] >= 2000:
                continue

            this_instance_lighter = list(x.iloc[i][lighter_columns].values)
            this_instance_heavier = list(x.iloc[i][heavier_columns].values)

            # adding weight
            this_instance_lighter.append(t.iloc[i].values[1])
            this_instance_heavier.append(t.iloc[i].values[2])

            # adding treatment, is_heavier
            this_instance_lighter.append(0)
            this_instance_heavier.append(1)

            # adding the outcome
            this_instance_lighter.append(y.iloc[i].values[1])
            this_instance_heavier.append(y.iloc[i].values[2])
            data.append(this_instance_lighter)
            data.append(this_instance_heavier)

        pd.DataFrame(columns=self.cols, data=data).to_csv(
            self.dataset_file, index=False
        )
        logging.info(f"saved preprocessed file at {self.dataset_file}")

    def load_raw_data_deprecated(self):
        logging.info("loading twins dataset")
        df = pd.read_csv(self.dataset_file).sample(frac=1)
        df.fillna(value=df.mean(), inplace=True)  # filling the missing values
        df.fillna(value=df.mode().loc[0], inplace=True)
        df1 = df.loc[df["treatment"] == 1].reset_index(drop=True)
        df0 = df.loc[df["treatment"] == 0].reset_index(drop=True)
        self.ground_truth_ate = df1["outcome"].mean() - df0["outcome"].mean()
        n1 = int(10000 * self.balance_param)
        n0 = 10000 - n1
        df1, df0 = df1.iloc[:n1], df0.iloc[:n0]
        df = pd.concat([df0, df1], ignore_index=True).sample(frac=1)
        self.context = df.loc[:, self.cols[:-2]].values
        self.action = df.loc[:, "treatment"].astype(int).values
        self.reward = df.loc[:, "outcome"].values
        self.position = None
        self.data = df
        self.outcome = "outcome"
        self.treatment = "treatment"
        self.context_cols = self.cols[:-2]
        self.reward_type = "continuous"

    def load_raw_data(self):
        logging.info("loading twins dataset")
        df = pd.read_csv(self.dataset_file)
        df.fillna(value=df.mean(), inplace=True)  # filling the missing values
        df.fillna(value=df.mode().loc[0], inplace=True)
        df1 = df.loc[df["treatment"] == 1].reset_index(drop=True)
        df0 = df.loc[df["treatment"] == 0].reset_index(drop=True)
        self.ground_truth_ate = df1["outcome"].mean() - df0["outcome"].mean()
        actions = np.random.binomial(n=1, p=(df1["gestat10"] / 10 - 0.1))
        df1_filtered = df1.loc[actions == 1].copy()
        df0_filtered = df0.loc[actions == 0].copy()
        df = pd.concat([df0_filtered, df1_filtered], ignore_index=True).sample(frac=1)
        self.context = df.loc[:, self.cols[:-2]].values
        self.action = df.loc[:, "treatment"].astype(int).values
        self.reward = df.loc[:, "outcome"].values
        self.position = None
        self.data = df
        self.outcome = "outcome"
        self.treatment = "treatment"
        self.context_cols = self.cols[:-2]
        self.reward_type = "continuous"


def get_ate_dataset(dataset_name, dataset_file, balance_param):
    if dataset_name == "ihdp":
        return IHDPDataset()
    elif dataset_name == "lalonde":
        return LalondeDataset()
    elif dataset_name == "twins":
        return TwinsDataset(dataset_file, balance_param)
    elif dataset_name == "criteo":
        return CriteoDataset(balance_param)
    else:
        raise ValueError(
            "Dataset name must be one of ['ihdp', 'lalonde', 'twins', 'criteo']"
        )
