import ast
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import wfdb


class PTBXLDataset:
    def __init__(self, data_path: str, sampling_rate: int = 100):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.database_path = os.path.join(data_path, "ptbxl_database.csv")
        self.scp_statements = os.path.join(data_path, "scp_statements.csv")

        self.df = None
        self.y = None
        self.label_encoders = []

    def load_metadata(self) -> pd.DataFrame:
        if not os.path.exists(self.database_path):
            raise FileNotFoundError(f"Database file not found at {self.database_path}")

        df = pd.read_csv(self.database_path, index_col="ecg_id")
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

        self.df = df
        return df

    def load_scp_statements(self) -> pd.DataFrame:
        if not os.path.exists(self.scp_statements):
            raise FileNotFoundError(
                f"SCP statements file not found at {self.scp_statements}"
            )

        scp_df = pd.read_csv(self.scp_statements, index_col=0)
        return scp_df

    def load_raw_data(self, df: pd.DataFrame, sampling_rate: int = 100) -> np.ndarray:
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        if sampling_rate == 100:
            data = [
                wfdb.rdsamp(os.path.join(self.data_path, f)) for f in df.filename_lr
            ]
        else:
            data = [
                wfdb.rdsamp(os.path.join(self.data_path, f)) for f in df.filename_hr
            ]

        data = np.array([signal for signal, meta in data])
        return data

    def prepare_labels(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[int], List[str]]:
        scp_statements = self.load_scp_statements()

        diagnostic_superclasses = (
            scp_statements[scp_statements.diagnostic == 1.0]["diagnostic_class"]
            .dropna()
            .unique()
            .tolist()
        )
        diagnostic_superclasses.sort()
        superclass_to_idx = {name: i for i, name in enumerate(diagnostic_superclasses)}

        y_labels = []
        valid_ecg_ids = []

        for ecg_id, row in df.iterrows():
            record_superclasses = (
                scp_statements.loc[
                    scp_statements.index.isin(row.scp_codes), "diagnostic_class"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            valid_record_superclasses = [
                sc for sc in record_superclasses if sc in diagnostic_superclasses
            ]

            if valid_record_superclasses:
                selected_superclass = valid_record_superclasses[0]
                y_labels.append(superclass_to_idx[selected_superclass])
                valid_ecg_ids.append(ecg_id)

        return (
            np.array(y_labels, dtype=np.int64),
            valid_ecg_ids,
            diagnostic_superclasses,
        )

    def split_data(
        self, stratify_col: str = "strat_fold", test_fold: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.df is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")

        test_df = self.df[self.df[stratify_col] == test_fold].copy()
        train_val_df = self.df[self.df[stratify_col] != test_fold].copy()

        train_df = train_val_df[train_val_df[stratify_col] <= 8].copy()
        val_df = train_val_df[train_val_df[stratify_col] == 9].copy()

        return train_df, val_df, test_df

    def get_dataset(self, test_fold: int = 10) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        List[str],
    ]:
        self.load_metadata()
        train_df, val_df, test_df = self.split_data(test_fold=test_fold)

        X_train = self.load_raw_data(train_df)
        X_val = self.load_raw_data(val_df)
        X_test = self.load_raw_data(test_df)

        y_train, valid_train_ids, class_names = self.prepare_labels(train_df)
        y_val, valid_val_ids, _ = self.prepare_labels(val_df)
        y_test, valid_test_ids, _ = self.prepare_labels(test_df)

        X_train = X_train[train_df.index.isin(valid_train_ids)]
        X_val = X_val[val_df.index.isin(valid_val_ids)]
        X_test = X_test[test_df.index.isin(valid_test_ids)]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names
