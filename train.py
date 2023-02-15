from sklearn import svm
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import hinge_loss
import pandas as pd


def train_linear_classifier(x_train, y_train, clf_name, vocab=None):
    if vocab:
        vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', binary=False, vocabulary=vocab)
    else:
        vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', binary=False)
    X_2_train = vectorizer.fit_transform(x_train).toarray()
    if clf_name == "SVM":
        clf = svm.LinearSVC(random_state=42, loss="hinge", max_iter=10000)
    clf.fit(X_2_train, y_train)
    return clf, vectorizer


def test_linear_classifier(x_test, y_test, clf, vectorizer):
    X_2_test = vectorizer.transform(x_test).toarray()
    score = clf.score(X_2_test, y_test)
    return score


def calc_clf_performance(df, neg, pos, size, iter_num):
    neg_df = df[df["scores"] == neg]
    pos_df = df[df["scores"] == pos]
    score_sum = 0
    for i in range(iter_num):
        neg_sample = neg_df.sample(n=2*size)
        pos_sample = pos_df.sample(n=2*size)
        neg_train = neg_sample.iloc[:size, :]
        neg_test = neg_sample.iloc[size:, :]
        pos_train = pos_sample.iloc[:size, :]
        pos_test = pos_sample.iloc[size:, :]
        train_df = pd.concat([neg_train, pos_train])
        test_df = pd.concat([neg_test, pos_test])
        clf, vectorizer = train_linear_classifier(train_df["reviews"].astype(str).tolist(), [0]*size + [1]*size, "SVM")
        score_sum += test_linear_classifier(test_df["reviews"].astype(str).tolist(), [0]*size + [1]*size, clf, vectorizer)
    return score_sum/iter_num


def train_and_test(year, category, neg, pos, size, iter_num, d):
    df_tmp = utils.get_star_by_year_df(category, year)
    d[year] = calc_clf_performance(df_tmp.head(100000), neg, pos, size, iter_num)
    print("Finished {}".format(year))


def sample_n_samples(year, category, size, iter_num, d, stars):
    samples = []
    df_tmp = utils.get_star_by_year_df(category, year, stars)
    for i in range(iter_num):
        sample = df_tmp.sample(n=2*size)
        text_sample = sample["reviews"].astype(str).tolist()
        samples.append(text_sample)
    d[year] = samples
    print("Finished {}".format(year))


def train_domain_classifier(year, category, size, iter_num, samples_dict, end_year, vocab, d, stars):
    total_years = end_year - year + 1
    a_distance_years = []
    for i in range(1, total_years):
        sum_a_distance = 0
        for j in range(iter_num):
            source = samples_dict[year][j]
            source_train = source[:size]
            source_test = source[size:]
            target = samples_dict[year + i][j]
            target_train = target[:size]
            target_test = target[size:]
            clf, vectorizer = train_linear_classifier(source_train + target_train, [0] * size + [1] * size, "SVM", vocab)
            X_test = vectorizer.transform(source_train + target_train).toarray()
            pred_decision = clf.decision_function(X_test)
            a_distance = hinge_loss([0] * size + [1] * size, pred_decision)
            sum_a_distance += 1 - a_distance
        a_distance_years.append(sum_a_distance/iter_num)
    d[year] = a_distance_years
    print("Finished {}".format(year))











