import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def get_class(name) :
    if name == "B" :
        return 0
    elif name == "H" :
        return 1
    elif name == "SD" :
        return 2
    elif name == "Se" :
        return 3
    elif name == "W" :
        return 4
    # indicate error
    else :
        return -1

def get_name(c) :
    if c == 0 :
        return "B"
    elif c == 1 :
        return "H"
    elif c == 2 :
        return "SD"
    elif c == 3 :
        return "Se"
    elif c == 4 :
        return "W"
    # indicate error
    else :
        return -1

def parse_input(lines) :
    features = []
    ids = []
    model = []
    answers = []
    for line in lines :
        if '@ATTRIBUTE' in line:
            if not ("id NUMERIC" in line or "location {B,H,SD,Se,W}" in line) :
                tokens = line.split(' ')
                features.append(tokens[1])
        elif '@RELATION' in line or '@DATA' in line :
            pass
        else :
            tokens = line.split(',')
            id = tokens[0]
            ids.append(int(id.strip()))
            answer = tokens[len(features) + 1]
            answers.append(int(get_class(answer.strip())))
            tokens.remove(id)
            tokens.remove(answer)
            tokens = [int(t) for t in tokens]
            model.append(tokens)
    return features, ids, model, answers

def print_predictions(ids, results, f) :
    for i in range(0, len(ids)) :
        print(str(ids[i]) + " : " + str(get_name(results[i])), file=f)

def print_stats(predicted, actual, f) :
    # 2D array representing confusion matrix
    stats = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    for i in range(0, len(actual)) :
        p = int(predicted[i])
        a = int(actual[i])
        stats[p][a] = stats[p][a] + 1
    # TP, FP, FN
    stats2 = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    # for each class
    for i in range(0, 5) :
        for j in range(0, 5) :
            if i == j :
                # true positive
                stats2[i][0] = stats2[i][0] + stats[i][j]
            else :
                # false positive for i
                stats2[i][1] = stats2[i][1] + stats[i][j]
                # false negative for j
                stats2[j][2] = stats2[j][2] + stats[i][j]
    print("Accuracy: " + str(accuracy_score(actual, predicted)) + '\n', file=f)
    for i in range(0, 5) :
        precision = stats2[i][0]/(stats2[i][0] + stats2[i][1])
        print("Precision - " + get_name(i) + ": " + str(precision), file=f)
        recall = stats2[i][0]/(stats2[i][0] + stats2[i][2])
        print("Recall - " + get_name(i) + ": " + str(recall), file=f)
        f1_score = (2 * precision * recall)/(precision + recall)
        print("F1 Score - " + get_name(i) + ": " + str(f1_score) + '\n', file=f)

def print_for_kaggle(ids, results, f) :
    print("Id,Category", file=f)
    for i in range(0, len(ids)) :
        print(str(ids[i]) + "," + str(get_name(results[i])), file=f)

def main() :
    # open training data file
    # NOTE: sys.argv[0] is the Python file itself, so we skip it
    train_file = open(sys.argv[1], 'r')
    dev_file = open(sys.argv[2], 'r')
    test_file = open(sys.argv[3], 'r')

    # create output files
    stats = open("stats.txt", "w")
    predictions = open("predictions.txt", "w")
    kaggle = open("dianaruth_kaggle.csv", "w")

    # parse ARFF format
    # 1. put features into an array
    # 2. create 2D array representing the vector space model for the tweets
    # 3. create array with classes for training data
    train_lines = train_file.readlines()
    dev_lines = dev_file.readlines()
    test_lines = test_file.readlines()

    # close files
    train_file.close()
    dev_file.close()
    test_file.close()

    # get parsed input
    train_features, train_ids, train_model, train_answers = parse_input(train_lines)
    dev_features, dev_ids, dev_model, dev_answers = parse_input(dev_lines)
    test_features, test_ids, test_model, test_answers = parse_input(test_lines)

    ###########################################################################
    # 1-NN
    ###########################################################################

    # set up classifier
    neighbors1 = KNeighborsClassifier(n_neighbors=1)
    neighbors1.fit(train_model, train_answers)

    # predict development data answers and calculate statistics
    dev_predicted_answers_1nn = []
    for m in dev_model :
        result = neighbors1.predict([m])
        dev_predicted_answers_1nn.append(result[0])
    print("-------------------------------------------------------------------", file=stats)
    print("-------------------------------------------------------------------", file=predictions)
    print("1-NN", file=stats)
    print("1-NN", file=predictions)
    print("-------------------------------------------------------------------", file=stats)
    print("-------------------------------------------------------------------", file=predictions)
    print_stats(dev_predicted_answers_1nn, dev_answers, stats)

    # predict test data answers
    test_predicted_answers_1nn = []
    for m in test_model :
        result = neighbors1.kneighbors([m])
        test_predicted_answers_1nn.append(result[0])
    print_predictions(test_ids, test_predicted_answers_1nn, predictions)

    ###########################################################################
    # 3-NN
    ###########################################################################

    # set up classifier
    neighbors3 = KNeighborsClassifier(n_neighbors=3)
    neighbors3.fit(train_model, train_answers)

    # predict development data answers and calculate statistics
    dev_predicted_answers_3nn = []
    for m in dev_model :
        result = neighbors3.predict([m])
        dev_predicted_answers_3nn.append(result[0])
    print("-------------------------------------------------------------------", file=stats)
    print("-------------------------------------------------------------------", file=predictions)
    print("3-NN", file=stats)
    print("3-NN", file=predictions)
    print("-------------------------------------------------------------------", file=stats)
    print("-------------------------------------------------------------------", file=predictions)
    print_stats(dev_predicted_answers_3nn, dev_answers, stats)

    # predict test data answers
    test_predicted_answers_3nn = []
    for m in test_model :
        result = neighbors3.predict([m])
        test_predicted_answers_3nn.append(result[0])
    print_predictions(test_ids, test_predicted_answers_3nn, predictions)

    ###########################################################################
    # Decision Tree
    ###########################################################################

    # set up classifier
    decision_tree = DecisionTreeClassifier()
    decision_tree = decision_tree.fit(train_model, train_answers)

    # predict development data answers and calculate statistics
    dev_predicted_answers_decision_tree = []
    for m in dev_model :
        result = decision_tree.predict([m])
        dev_predicted_answers_decision_tree.append(result[0])
    print("-------------------------------------------------------------------", file=stats)
    print("-------------------------------------------------------------------", file=predictions)
    print("Decision Tree", file=stats)
    print("Decision Tree", file=predictions)
    print("-------------------------------------------------------------------", file=stats)
    print("-------------------------------------------------------------------", file=predictions)
    print_stats(dev_predicted_answers_decision_tree, dev_answers, stats)

    # predict test data answers
    test_predicted_answers_decision_tree = []
    for m in test_model :
        result = decision_tree.predict([m])
        test_predicted_answers_decision_tree.append(result[0])
    print_predictions(test_ids, test_predicted_answers_decision_tree, predictions)

    ###########################################################################
    # Naive Bayes
    ###########################################################################

    # set up classifier
    gnb = GaussianNB()
    nb = gnb.fit(train_model, train_answers)

    # predict development data answers and calculate statistics
    dev_predicted_answers_nb = []
    for m in dev_model :
        result = nb.predict([m])
        dev_predicted_answers_nb.append(result[0])
    print("-------------------------------------------------------------------", file=stats)
    print("-------------------------------------------------------------------", file=predictions)
    print("Naive Bayes", file=stats)
    print("Naive Bayes", file=predictions)
    print("-------------------------------------------------------------------", file=stats)
    print("-------------------------------------------------------------------", file=predictions)
    print_stats(dev_predicted_answers_nb, dev_answers, stats)

    # predict test data answers
    test_predicted_answers_nb = []
    for m in test_model :
        result = nb.predict([m])
        test_predicted_answers_nb.append(result[0])
    print_predictions(test_ids, test_predicted_answers_nb, predictions)
    # print_for_kaggle(test_ids, test_predicted_answers_nb, kaggle)

    # close output files
    stats.close()
    predictions.close()
    kaggle.close()

if __name__ == '__main__' :
    main()
