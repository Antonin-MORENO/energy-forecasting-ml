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
        
        # Identify and sort all lag feature columns (e.g., "I014_ND_lag_X)
        self.lag_cols = sorted(
            [c for c in self.feature_cols if c.startswith(f"{self.target_col}_lag_")],
            key=lambda x: int(x.split('_')[-1])
        )
        
        # All remaining feature columns are treated as static (non-lagged) inputs
        self.stat_cols = [c for c in self.feature_cols if c not in self.lag_cols]


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
    
    
    def create_sequences(self, df: pd.DataFrame):
        """
        Build sequence and static inputs and target for sequence models:
        - X_seq: (n_samples, timesteps, 1)
        - X_stat: (n_samples, n_static) or None
        - y:      (n_samples,)
        """
        # Extract lagged features as a 2D array (n_samples, timesteps)
        X_lags = df[self.lag_cols].values
        
        # Determine the number of samples and the sequence length
        n_samples, timesteps = X_lags.shape
        
        # Reshape into a 3D tensor for sequence models: (n_samples, timesteps, 1)
        X_seq = X_lags.reshape(n_samples, timesteps, 1)

        # Extract static (non-lagged) features
        X_stat = df[self.stat_cols].values if self.stat_cols else None

        # Extract the target values as a 1D array
        y = df[self.target_col].values

        return X_seq, X_stat, y
    
    def get_sequence_data(self, val_years: int = 1):
        
        """
        Splits data into training, validation, and test sets, then creates sequences.

        - test: The final `self.holdout_years` of the dataset.
        - trainval: The remaining data, used to fit the scaler.
        - val: The final `val_years` from the scaled trainval set.
        - train: The rest of the trainval set.

        Returns:
            X_seq_tr, X_stat_tr, y_tr,
            X_seq_val, X_stat_val, y_val,
            X_seq_te, X_stat_te, y_te
        """
        # Load full dataset and split into train/test by time
        df = self.load_data()
        df_trainval, df_test = self.temporal_split(df)

        # Scale features on train/test, returning numpy arrays for features and targets
        X_trval_arr, y_trval, X_te_arr, y_te = self.scale_split(df_trainval, df_test)

        # Rebuild scaled train+val DataFrame and reattach dates
        df_trval_s = pd.DataFrame(X_trval_arr, columns=self.feature_cols)
        df_trval_s[self.target_col] = y_trval
        df_trval_s[self.date_col]   = df_trainval[self.date_col].values

        
        # Rebuild scaled test DataFrame and reattach dates
        df_test_s = pd.DataFrame(X_te_arr, columns=self.feature_cols)
        df_test_s[self.target_col] = y_te
        df_test_s[self.date_col]   = df_test[self.date_col].values

        # Split off the last `val_years` years for validation
        dates = pd.to_datetime(df_trval_s[self.date_col])
        val_start = dates.max() - pd.DateOffset(years=val_years)
        mask_val = dates >= val_start

        df_val_s   = df_trval_s[mask_val].reset_index(drop=True)
        df_train_s = df_trval_s[~mask_val].reset_index(drop=True)

        
        # Drop date column before sequence creation
        for d in (df_train_s, df_val_s, df_test_s):
            d.drop(columns=[self.date_col], inplace=True)

        # Convert each DataFrame into:
        #  - X_seq: lagged sequences tensor (n_samples, timesteps, 1)
        #  - X_stat: static features array (n_samples, n_static) or None
        #  - y_seq: target array (n_samples,)
        X_seq_tr,  X_stat_tr,  y_tr_seq  = self.create_sequences(df_train_s)
        X_seq_val, X_stat_val, y_val_seq = self.create_sequences(df_val_s)
        X_seq_te,  X_stat_te,  y_te_seq  = self.create_sequences(df_test_s)

        return X_seq_tr, X_stat_tr, y_tr_seq, X_seq_val, X_stat_val, y_val_seq, X_seq_te, X_stat_te, y_te_seq
            
    

    def get_train_val_test_split(self, val_years: int = 1):
        """
        Splits data into training, validation, and test sets for traditional ML models.

        - 1. Splits the full dataset into a preliminary train/val set and a final test set.
        - 2. Splits the preliminary train/val set further into a final training set and a validation set.
        - 3. Fits the scaler ONLY on the final training set and transforms all three sets.
        
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test) as NumPy arrays.
        """
        # Load and split the full dataset into train/val and test sets
        df = self.load_data()
        df_trainval, df_test = self.temporal_split(df)

        
        
        # Determine the cutoff date to separate the training set from the validation set.
        # This is done by taking the last date in the combined train/val set and subtracting the specified number of years.
        val_cutoff = df_trainval[self.date_col].max() - pd.DateOffset(years=val_years)
        
        # Create the final training and validation dataframes based on the cutoff date.
        df_train = df_trainval[df_trainval[self.date_col] < val_cutoff].copy()
        df_val = df_trainval[df_trainval[self.date_col] >= val_cutoff].copy()

        print(f"Train set: {df_train[self.date_col].min()} to {df_train[self.date_col].max()}")
        print(f"Validation set: {df_val[self.date_col].min()} to {df_val[self.date_col].max()}")
        print(f"Test set: {df_test[self.date_col].min()} to {df_test[self.date_col].max()}")

        # Identify the columns that require scaling by excluding specified columns.
        cols_to_scale = [col for col in self.feature_cols if col not in self.no_scale_cols]

        # Initialize the scaler based on the specified type ('standard' or 'minmax').
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        # Prepare the feature dataframes for each set.
        X_train_df = df_train[self.feature_cols].copy()
        X_val_df = df_val[self.feature_cols].copy()
        X_test_df = df_test[self.feature_cols].copy()

        # Extract the target variable arrays from each set.
        y_train = df_train[self.target_col].values
        y_val = df_val[self.target_col].values
        y_test = df_test[self.target_col].values

        # If a scaler is initialized, fit it ONLY on the training data to avoid data leakage.
        # Then, use the fitted scaler to transform the training, validation, and test sets.
        if self.scaler is not None:
            X_train_df[cols_to_scale] = self.scaler.fit_transform(X_train_df[cols_to_scale])
            X_val_df[cols_to_scale] = self.scaler.transform(X_val_df[cols_to_scale])
            X_test_df[cols_to_scale] = self.scaler.transform(X_test_df[cols_to_scale])

        # Return the six required NumPy arrays for model training and evaluation.
        return X_train_df.values, y_train, X_val_df.values, y_val, X_test_df.values, y_test