import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

class DataHandler:
    """
    Class to load, temporally split, and scale the dataset.
    """

    def __init__(self,
                 csv_path: str,
                 date_col: str,
                 feature_cols: list,
                 target_col: str,
                 holdout_ratio: float = 0.1,
                 scaler_type: str = 'standard'):
        self.csv_path = csv_path
        self.date_col = date_col
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.holdout_ratio = holdout_ratio
        self.scaler_type = scaler_type
        self.scaler = None

    def load_data(self):
        """
        Load the CSV while keeping the date_col in df.columns,
        convert date_col to datetime, sort by date_col, and return df.
        """
        df = pd.read_csv(
            self.csv_path,
            parse_dates=[self.date_col]  # convert the specified column to datetime
        )
        # Sort by the date_col to ensure chronological order
        df = df.sort_values(self.date_col).reset_index(drop=True)
        return df

    def temporal_split(self, df: pd.DataFrame):
        """
        Simple temporal split:
          - keep the last self.holdout_ratio % of rows for the final test set
          - use the rest for training/validation
        Returns (df_trainval, df_test), both with the date_col still present.
        """
        n_total = len(df)
        n_holdout = int(self.holdout_ratio * n_total)
        n_trainval = n_total - n_holdout

        df_trainval = df.iloc[:n_trainval].copy().reset_index(drop=True)
        df_test     = df.iloc[n_trainval:].copy().reset_index(drop=True)
        return df_trainval, df_test

    def scale_split(self,
                    df_trainval: pd.DataFrame,
                    df_test: pd.DataFrame):
        """
        Scale the feature columns (only on trainval), then apply the same transform to the test set.
        Returns numpy arrays: (X_trainval, y_trainval, X_test, y_test).
        """
        # Extract feature and target values
        X_trval = df_trainval[self.feature_cols].values
        y_trval = df_trainval[self.target_col].values
        X_test  = df_test[self.feature_cols].values
        y_test  = df_test[self.target_col].values

        # Choose the scaler type
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        # Fit the scaler on trainval, then transform both trainval and test
        if self.scaler is not None:
            X_trval = self.scaler.fit_transform(X_trval)
            X_test  = self.scaler.transform(X_test)

        return X_trval, y_trval, X_test, y_test

    def get_time_series_cv(self, n_splits: int = 5):
        """
        Create and return a TimeSeriesSplit object with n_splits folds.
        """
        return TimeSeriesSplit(n_splits=n_splits)