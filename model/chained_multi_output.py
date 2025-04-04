from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.multioutput import ClassifierChain
from model.randomforest import RandomForest
from sklearn.preprocessing import MultiLabelBinarizer


def multi_label_binarization(df, columns):
    """Multi-label binarization and ensures input columns are iterable.

    arguments are:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): A list of column names in the DataFrame what we want to binarize.

    returns:
    np.ndarray: (np.hstack(Y_transformed) Concatenated result of the binarization.
    y2, y3, y4 are one-hot encoded
    mlb_list: A list of MultiLabelBinarizer instances used for each column. Shows which labels
    corresponding to the columns in the binarized matrix.
     """
    mlb_list = [MultiLabelBinarizer() for _ in columns]

    # Ensure all entries in the specified columns are iterable
    for col in columns:
        df[col] = df[col].apply(lambda x: x if isinstance(x, (list, set)) else ([x] if pd.notnull(x) else []))

    # Transform the columns using MultiLabelBinarizer and return the concatenated target matrix
    Y_transformed = [mlb.fit_transform(df[col]) for mlb, col in zip(mlb_list, columns)]
    return np.hstack(Y_transformed), mlb_list


def train_classifier_chain(X_train, Y_train):
    """ClassifierChain model training.
    arguments are:
    X_train: The feature matrix for train
    Y_train: The labels

    return:
    chain_clf: The trained Classifier
    """
    # Dynamically determine the number of labels for the chain order
    # Y_train: an array where the columns are the labels.

    num_labels = Y_train.shape[1]  # Number of multi-labels
    chain_order = list(range(num_labels))  # Order must cover all labels dynamically

    # Using the RandomForest class from the randomforest.py to create a base model.

    base_model = RandomForest(
        model_name="Chained_Model",
        embeddings=None,  # Embeddings are not directly needed here; training data is used instead
        y=None  # Multi-label data will be provided via ClassifierChain
    ).mdl  # Access the underlying RandomForestClassifier instance

    # Initialize the ClassifierChain with the dynamically determined order
    chain_clf = ClassifierChain(base_model, order=chain_order)

    # Train the model
    chain_clf.fit(X_train, Y_train)
    return chain_clf


def binarize_and_split_data(df, label_columns, embeddings, test_size=0.2):
    """"" Multi-label binarization and train-test splitting.
    arguments:
    df: containing the data, label_columns: list of the column names to binarize,
    embeddings: the feature matrix, test_size: The part of the data which is used for testing.

    returns:
    X_train, X_test: features
    Y_train, Y_test: binarized labels for train and test
    mlb_list: MultiLabelBinarizer objects for mapping the decoded labels back
    """
    #Multi-label  binarization
    Y, mlb_list = multi_label_binarization(df, label_columns)
    #Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(
        embeddings, Y, test_size=test_size, random_state=42)
    return X_train, X_test, Y_train, Y_test, mlb_list


def evaluate_classifier_chain(chain_clf, X_test, Y_test, label_columns, mlb_list):
    """Evaluates the classifier chain and generates a classification report.
    arguments:
    chain_clf: the trained model to evaluate
    X_test: the feature matrix for testing
    Y_test: the binarized labels for testing
    label_columns: column names of the labels
    mlb_list: MultiLabelBinarizer objects for mapping the decoded labels back
    """
    # generate the predictions
    Y_pred = chain_clf.predict(X_test)

    # evaluate the results
    print("Classifier Chain Model Results:")

    start_idx = 0 # We are starting with 0th element and going through the labels
    for col_idx, label_column in enumerate(label_columns):
        print(f"\n### Classification Report for {label_column} ###")

        # Calculate the number of classes for the current column
        num_classes = len(mlb_list[col_idx].classes_)

        # Slice the binarized labels from Y_test and Y_pred for current label
        Y_test_col = Y_test[:, start_idx:start_idx + num_classes]
        Y_pred_col = Y_pred[:, start_idx:start_idx + num_classes]

        # Update the start index for the next label
        start_idx += num_classes

        # Reversing the multi-label binarization to get back the original labels
        Y_true_unbinarized = mlb_list[col_idx].inverse_transform(Y_test_col)
        Y_pred_unbinarized = mlb_list[col_idx].inverse_transform(Y_pred_col)

        # Convert to single-label format for compatibility with classification_report
        true_flat = [";".join(sorted(labels)) for labels in Y_true_unbinarized]
        pred_flat = [";".join(sorted(labels)) for labels in Y_pred_unbinarized]

        # Generate and print the Classification Report (sckit-learn)
        report = classification_report(
            true_flat,
            pred_flat,
            zero_division=1  # Suppress warnings for labels with no true samples
        )
        print(report)


def run_second_model(data, df, label_columns):
    """Executes the Second Model pipeline. (Chained Multi-output with Random Forest)
    data: the datasets we will pass to the embeddings function, which will convert
    the text to numerical vectors.
    df: the DataFrame containing the data, including the multi-label columns that
    will be binarized.
    label_columns: the list of column names in the DataFrame that we want to binarize.
    """
    print("Chained Multi-output with Random Forest")

    embeddings = data.get_embeddings() #fetch embeddings

    # Binarize and split data
    X_train, X_test, Y_train, Y_test, mlb_list = binarize_and_split_data(df, label_columns, embeddings)

    # Train ClassifierChain
    chain_clf = train_classifier_chain(X_train, Y_train)

    # Evaluate Results
    evaluate_classifier_chain(chain_clf, X_test, Y_test, label_columns, mlb_list)