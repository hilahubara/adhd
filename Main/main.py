import collections
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

methods_dict = {'svc': LinearSVC(),
                'DecisionTree': DecisionTreeClassifier(),
                'rf': RandomForestClassifier(n_estimators=100, n_jobs=os.cpu_count()),
                'lr': LogisticRegression(solver='lbfgs', n_jobs=os.cpu_count())}


def write_results(results, results_path):
    """
    Write the results to a file

    """
    print(f'{datetime.now()} >> Writing results...')
    if not os.path.isdir(results_path):
        try:
            os.mkdir(results_path)
        except FileNotFoundError:
            results_path = get_root_path() + '\\Results'
            if not os.path.isdir(results_path):
                os.mkdir(results_path)

    path = results_path + '\\results.csv'

    try:
        if not os.path.isfile(path):
            with open(results_path + '\\results.csv', mode='w+', newline='') as result_file:
                result_writer = csv.writer(result_file)
                result_writer.writerow(
                    ['UID.PM', 'ML Meth.', 'accuracy', 'macro avg', '', '', '', 'micro avg', '', '', ''])
                result_writer.writerow(
                    ['', '', '', 'precision', 'recall', 'f1-score', 'support', 'precision', 'recall', 'f1-score',
                     'support'])

        with open(results_path + '\\results.csv', mode='a', newline='') as result_file:
            result_writer = csv.writer(result_file)
            for id_, result in results.items():
                for method, scores in result.items():
                    values = list()
                    for score, val in scores.items():
                        if score in ['micro avg', 'macro avg', 'accuracy']:
                            values.extend(val.values())

                    result_writer.writerow([id_, method] + values)

    except PermissionError:
        print('results file must be closed when the program runs')
        raise SystemExit(0)


def get_root_path():
    """
    Returns the path to the root folder of the project
    """
    path = str(Path(__file__).parent.parent)
    return path


def read_config(key: str):
    """
    Reading from the config file defined by the user.
    The config file must be in the Main folder and named config
    :param key: A key that appears in the json file
    :return: The value corresponding to the key
    """
    try:
        with open('config.json') as json_file:
            data = json.load(json_file)
            val = data[key]
            return val
    except FileNotFoundError:
        print('config file does not exists. look at README how to fix it')
        raise SystemExit(0)


def read_data(dataset_path) -> List[str]:
    """
    Reads the data according to the path and returns
    a list of str that contains all dataset rows

    """
    print(f'{datetime.now()} >> Start Reading from DataSet...')
    if not os.path.isfile(dataset_path):
        dataset_path = get_root_path() + '\\DataSet\\dataset.csv'
    with open(dataset_path, mode='r', encoding='utf-8') as dataset:
        data = dataset.readlines()[1:]
    return data


def extract_data(original_data, columns_ignore, test_columns, uid_column):
    """
    Filters the irrelevant columns and returns the data as a collection of dictionaries
    as well as a list of labels (have ADHD or not) and a list of profile IDs
    :param original_data: The original data
    :param columns_ignore: The columns to be omitted from the data
    :param test_columns: The column on which we will take the test
    :param uid_column: The column with the ID number
    """
    columns_ignore.insert(len(columns_ignore), test_columns)
    data, labels, groups = list(), list(), list()

    for line in original_data:
        line = line.replace('\n', '').split(',')
        new_line = {i: line[i] for i in range(len(line)) if i not in columns_ignore}
        data.append(new_line)
        labels.append(line[test_columns])
        groups.append(line[uid_column])

    return data, labels, groups


def extract_features(data):
    """
    Converts data to matrix
    :param data: The data we want to perform the Cross-Validation
    :return: A matrix that represents the data
    """
    vectorizer = DictVectorizer()
    print(f'{datetime.now()} >> Extracting Features...')
    features = vectorizer.fit_transform(data)
    return features


def main():
    start = datetime.now()

    # getting parameters from config file
    dataset_path = read_config('dataset_path')
    ignored_columns = read_config('ignored_columns')
    test_columns = read_config('test_columns')
    uid_column = read_config('uid_column')
    methods = read_config('methods')
    results_path = read_config('results_path')

    # reading the dataset file שnd creating the appropriate matrix
    data = read_data(dataset_path)
    data, labels, groups = extract_data(data, ignored_columns, test_columns, uid_column)
    features = extract_features(data)

    # Converting words to numbers
    le = LabelEncoder()
    le.fit(labels + groups)
    labels = le.transform(labels)
    groups = le.transform(groups)

    # Create Instance of Cross-Validation
    leave_one_out = LeaveOneGroupOut()
    leave_one_out = leave_one_out.split(X=features, y=labels, groups=groups)

    # Division the data into training and testing according to the Cross-Validation defined above
    results = dict()
    for train_index, test_index in leave_one_out:
        result = dict()
        profile_id = le.inverse_transform(groups[test_index])[0]
        print(f'{datetime.now()} >> Profile {profile_id} is selected for the test')
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Training, prediction and writing results to a file
        for method in methods:
            print(f'{datetime.now()} >> Running {method}...')
            method_ = methods_dict[method]
            method_.fit(X_train, y_train)
            y_pred = method_.predict(X_test)
            report = classification_report(y_true=y_test, y_pred=y_pred,
                                           output_dict=True)
            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            report['accuracy'] = {'accuracy': acc}
            report = dict(collections.OrderedDict(sorted(report.items())))
            result[method] = (report)
        results[profile_id] = result

    # Writing results
    write_results(results, results_path)
    print(f'{datetime.now()} >> Results were written successfully')
    print(f'{datetime.now()} >> Program Run Time: {datetime.now() - start}')


if __name__ == '__main__':
    main()
