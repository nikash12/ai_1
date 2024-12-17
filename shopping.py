import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Test size for train-test split
TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE, random_state=42
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    Evidence should be a list of lists, where each list contains the following:
        - Administrative: an integer
        - Administrative_Duration: a floating point number
        - Informational: an integer
        - Informational_Duration: a floating point number
        - ProductRelated: an integer
        - ProductRelated_Duration: a floating point number
        - BounceRates: a floating point number
        - ExitRates: a floating point number
        - PageValues: a floating point number
        - SpecialDay: a floating point number
        - Month: an index from 0 (January) to 11 (December)
        - OperatingSystems: an integer
        - Browser: an integer
        - Region: an integer
        - TrafficType: an integer
        - VisitorType: 0 (not returning) or 1 (returning)
        - Weekend: 0 (if false) or 1 (if true)

    Labels should be a list where each label is 1 if Revenue is true, 0 otherwise.
    """
    # Map month abbreviations to numbers
    month_mapping = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }

    evidence = []
    labels = []

    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                month_mapping[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0,
            ])
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return (evidence, labels)

def train_model(evidence, labels):
    """
    Train a k-nearest neighbor model (k=1) on the evidence and labels.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Evaluate the model's sensitivity and specificity.

    Sensitivity (True Positive Rate): Proportion of actual positives correctly identified.
    Specificity (True Negative Rate): Proportion of actual negatives correctly identified.
    """
    # Initialize counts
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    # Calculate counts
    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            if predicted == 1:
                true_positive += 1
            else:
                false_negative += 1
        elif actual == 0:
            if predicted == 0:
                true_negative += 1
            else:
                false_positive += 1

    # Calculate sensitivity and specificity
    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
