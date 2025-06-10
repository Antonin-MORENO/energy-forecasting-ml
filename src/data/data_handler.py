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
                 no_scale_cols: list = None,
                 holdout_ratio: float = 0.1,
                 holdout_years: int = None,
                 scaler_type: str = 'standard'):
        self.csv_path = csv_path
        self.date_col = date_col
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.no_scale_cols = no_scale_cols
        self.holdout_ratio = holdout_ratio
        self.holdout_years  = holdout_years
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
        Temporal split of the dataset:
        - If holdout_years is specified, keep the last N years for the test set
        - Otherwise, keep the last self.holdout_ratio % of rows for the final test set
        - Use the remaining rows for training/validation
        Returns (df_trainval, df_test), both with the date_col still present.
        """
        
        # Split by years
        if self.holdout_years is not None:
            # Find the maximum date, then subtract N years to get the cutoff
            max_date = df[self.date_col].max()
            cutoff   = max_date - pd.DateOffset(years=self.holdout_years)


            df_trainval = df[df[self.date_col] < cutoff].copy().reset_index(drop=True)  # All rows with date < cutoff
            df_test     = df[df[self.date_col] >= cutoff].copy().reset_index(drop=True) #  All rows ≥ cutoff 
            return df_trainval, df_test
        
        # Split by percentage 
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
        Scale a subset of feature columns (only on trainval), then apply the same transform
        to the test set, while keeping other features untouched.
        Returns numpy arrays: (X_trainval, y_trainval, X_test, y_test).
        """
        # Identify columns to scale by excluding those in no_scale_cols
        cols_to_scale = [col for col in self.feature_cols if col not in self.no_scale_cols]

        # Extract target data (unchanged)
        y_trval = df_trainval[self.target_col].values
        y_test = df_test[self.target_col].values

        
        # Prepare the complete feature DataFrames
        X_trval_df = df_trainval[self.feature_cols].copy()
        X_test_df = df_test[self.feature_cols].copy()
        
        
        # Choose the scaler type
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        # Fit the scaler on trainval, then transform both trainval and test
        if self.scaler is not None:
            X_trval_df[cols_to_scale] = self.scaler.fit_transform(X_trval_df[cols_to_scale])
            X_test_df[cols_to_scale] = self.scaler.transform(X_test_df[cols_to_scale])
            
            
        return X_trval_df.values, y_trval, X_test_df.values, y_test

    def get_time_series_cv(self, n_splits: int = 5):
        """
        Create and return a TimeSeriesSplit object with n_splits folds.
        """
        return TimeSeriesSplit(n_splits=n_splits)