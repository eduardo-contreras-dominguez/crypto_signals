import pandas as pd
from sklearn.model_selection import train_test_split
class Splitter:
    """
    A class to split a DataFrame into training and testing sets randomly.
    """

    @staticmethod
    def split_data(features: pd.DataFrame, target: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """
        Split the data into training and testing sets randomly.

        Args:
        - features (pd.DataFrame): The features of the data to be used for training.
        - target (pd.Series): The target variable for the prediction.
        - test_size (float): The proportion of the data to be used for testing.
        - random_state (int): The seed for random number generation, for reproducibility.

        Returns:
        - X_train (pd.DataFrame): Features for the training set.
        - X_test (pd.DataFrame): Features for the testing set.
        - y_train (pd.Series): Target variable for the training set.
        - y_test (pd.Series): Target variable for the testing set.
        """
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, y_train, y_test