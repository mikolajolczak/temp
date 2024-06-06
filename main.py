import csv
import time
import argparse
from sklearn.metrics import classification_report
from tqdm import tqdm

from LazyClassificationwithContrastPatterns import LazyClassificationwithContrastPatterns
from NaiveBayesianClassifier import NaiveBayesianClassifier
from SPRINT import SPRINT
from DatasetParser import DatasetParser

FLOAT= 'float'
INTEGER = 'int'
STRING = 'string'

TRAINING_SPLIT_RATIO = 0.8
NUMBER_OF_BUCKETS = 5

def train_classifier(classifier, training_data, column_info):
    classifier.fit(training_data, column_info)

def evaluate_classifier(classifier, test_data):
    true_labels, predicted_labels = [], []
    for row in tqdm(test_data):
        sample_features, ground_truth = row[:-1], row[-1]
        predicted_label = classifier.predict(sample_features)
        true_labels.append(ground_truth)
        predicted_labels.append(predicted_label)
    return calculate_metrics(true_labels, predicted_labels)

def calculate_metrics(true_labels, predicted_labels):
    return classification_report(true_labels, predicted_labels, zero_division=0)

def display_results(results_report, dataset_name, classifier, elapsed_time):
    print(f'Classifier: {classifier.__class__.__name__}')
    print(f'Dataset: {dataset_name}.csv')
    print(f'Time: {round(elapsed_time, 2)} seconds')
    print(results_report)

def parse_dataset(file_path, train_ratio, buckets):
    with open(file_path) as file:
        csv_data = list(csv.reader(file, delimiter=','))
        num_elements_first_row = len(csv_data[0])
        filtered_csv_data = [row for row in csv_data if len(row) == num_elements_first_row]
        column_data_types = determine_column_types(filtered_csv_data)
        return DatasetParser(filtered_csv_data, train_ratio, column_data_types, buckets)
def determine_column_types(csv_data):
    column_types = []
    header = csv_data[0]

    for i in range(len(header)):
        column = [row[i] for row in csv_data]
        column_set = set(column)
        if all(is_numeric(value) for value in column_set):
            column_types.append(FLOAT)
        elif all(is_integer(value) for value in column_set):
            column_types.append(INTEGER)
        else:
            column_types.append(STRING)
    return column_types

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def is_integer(value):
    try:
        return float(value).is_integer()
    except ValueError:
        return False

def split_dataset_for_classifier(dataset, classifier_name):
    if classifier_name in ["NaiveBayesianClassifier", "SPRINT"]:
        return dataset.get_training_set(binned=True), dataset.get_test_set(binned=True)
    else:
        return dataset.get_training_set(binned=False), dataset.get_test_set(binned=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classification algorithms on datasets.')
    parser.add_argument('dataset_files', nargs='+', help='Paths to dataset CSV files')
    
    args = parser.parse_args()
    
    for dataset_file in args.dataset_files:
        dataset_name = dataset_file.split('/')[-1].split('.')[0]
        dataset = parse_dataset(dataset_file, TRAINING_SPLIT_RATIO, NUMBER_OF_BUCKETS)
        classifiers = [
            NaiveBayesianClassifier(0),
            NaiveBayesianClassifier(0.1),
            NaiveBayesianClassifier(0.5),
            NaiveBayesianClassifier(1),
            SPRINT(),
            LazyClassificationwithContrastPatterns()
        ]
        for classifier in classifiers:
            training_set, testing_set = split_dataset_for_classifier(dataset, classifier.__class__.__name__)
            start_time = time.time()
            train_classifier(classifier, training_set, column_info=dataset.get_metadata())
            results = evaluate_classifier(classifier, testing_set)
            elapsed_time = time.time() - start_time
            display_results(results, dataset_name, classifier, elapsed_time)
