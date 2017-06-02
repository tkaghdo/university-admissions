"""
university_admissions.py is used to predict university admissions using logistic regression and generate some basic
information about the model
usage:
python university_admissions.py
"""

__author__ = "Tamby Kaghdo"

import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def main():
    try:
        # import the data
        admissions_df = pd.read_csv("data/admissions.csv")
    except Exception as e:
        print(e)
        sys.exit(1)

    print(admissions_df.head())

    # fit the logistic regression model. use gpa to predict admissions
    lr = LogisticRegression()
    lr.fit(admissions_df[["gpa"]], admissions_df["admit"])

    prediction_probs = lr.predict_proba(admissions_df[["gpa"]])

    # Probability that the row belongs to label `0`.
    print("Probability that the row belongs to label `0`")
    print(prediction_probs[:,0])
    # Probability that the row belongs to label `1`.
    print("Probability that the row belongs to label `1`")
    print(prediction_probs[:,1])

    # plot the probabilities
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.scatter(admissions_df["gpa"] ,prediction_probs[:,0], color="green")
    ax2.scatter(admissions_df["gpa"] ,prediction_probs[:,1], color="blue")
    ax1.set_ylabel("Probability that a student is not admitted")
    ax1.set_xlabel("GPA")
    ax2.set_ylabel("Probability that a student is admitted")
    ax2.set_xlabel("GPA")
    plt.tight_layout()
    #plt.show()

    # create predictions
    predictions = lr.predict(admissions_df[["gpa"]])
    print("*** Predictions ***")
    print(predictions)

    # plot the predictions
    plt.scatter(admissions_df["gpa"], predictions)
    plt.xlabel("GPA")
    plt.ylabel("Predictions")
    #plt.show()

    admissions_df["admit_predictions"] = predictions
    print(admissions_df.head())

    # predictions count
    print(admissions_df["admit_predictions"].value_counts())

    # *** calculate accuracy ***

    # get the rows where the actual and the predicted labels match
    matched_df = admissions_df[admissions_df["admit"] == admissions_df["admit_predictions"]]
    accuracy = float(len(matched_df)) / float(len(admissions_df))
    print("Accuracy is {0}".format(accuracy))

    # *** calculate the outcomes of the binary classification
    true_positives = len(admissions_df[(admissions_df["admit"] == 1) & (admissions_df["admit_predictions"] == 1)])
    true_negatives = len(admissions_df[(admissions_df["admit"] == 0) & (admissions_df["admit_predictions"] == 0)])
    false_positives = len(admissions_df[(admissions_df["admit"] == 0) & (admissions_df["admit_predictions"] == 1)])
    false_negatives = len(admissions_df[(admissions_df["admit"] == 1) & (admissions_df["admit_predictions"] == 0)])

    print("True Positives is {0}".format(true_positives))
    print("True Negatives is {0}".format(true_negatives))
    print("False Positives is {0}".format(false_positives))
    print("False Negatives is {0}".format(false_negatives))

    sensitivity = float(true_positives) / float((true_positives + false_negatives))
    print("Sensitivity is {0}".format(sensitivity))

    specificity = float(true_negatives) / float(true_negatives + false_positives)
    print("Specificity is {0}".format(specificity))

if __name__ == "__main__":
    sys.exit(0 if main() else 1)