from matplotlib import pyplot as plt
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from multiprocess import Process, Manager, Pool
import utils
from vader import SentimentIntensityAnalyzer
import os


neg_rating = 1


def get_histogram(input_list, x_name, y_name, title, category):
    bins = np.linspace(math.ceil(min(input_list)), math.floor(max(input_list)), 40)  # fixed number of bins
    plt.clf()
    plt.xlim([min(input_list) - 5, max(input_list) + 5])
    plt.hist(input_list, bins=bins, alpha=0.5)
    plt.title("{} - {}".format(title, category))
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig("{} - {}".format(title, category))


def calc_avg_sentiment(df, lexicon, use_abs=None):
    vocab, score_arr = create_vocab(lexicon, use_abs)
    vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', vocabulary=vocab, binary=False)
    feature_matrix = vectorizer.fit_transform(df['reviews'].values.astype('U')).toarray()
    sent = feature_matrix.dot(score_arr)
    norm_sent = []
    not_in_use = 0
    for i in range(sent.shape[0]):
        number_of_words = sum(feature_matrix[i])
        if number_of_words > 0:
            norm_sent.append(sent[i]/(sum(feature_matrix[i])))
        else:
            not_in_use += 1
    result = sum(norm_sent) / (len(norm_sent)-not_in_use)
    std = np.std(norm_sent)
    return result, std


def calc_sen(year, category, sample_size, lexicon, task_dict, std_dict, stars):
    df_tmp = utils.get_star_by_year_df(category, year, stars)
    task_dict[year],  std_dict[year] = calc_avg_sentiment(df_tmp.sample(sample_size, replace=True), lexicon)
    print("Finished {}".format(year))


def calc_abs(year, category, sample_size, lexicon, task_dict, std_dict, stars):
    df_tmp = utils.get_star_by_year_df(category, year, stars)
    task_dict[year], std_dict[year] = calc_avg_sentiment(df_tmp.sample(sample_size, replace=True), lexicon, use_abs=True)
    print("Finished {}".format(year))


def calc_avg_lexicon_coverage(df, lexicon):
    vocab, score_arr = create_vocab(lexicon)
    vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', vocabulary=vocab, binary=False)
    feature_matrix = vectorizer.fit_transform(df['reviews'].values.astype('U')).toarray()
    in_use = 0
    for i in range(feature_matrix.shape[0]):
        number_of_words = sum(feature_matrix[i])
        if number_of_words > 0:
            in_use += 1
    result = (in_use / feature_matrix.shape[0]) * 100
    return result


def calc_lexicon_coverage(year, category, lexicon, d, stars):
    df_tmp = utils.get_star_by_year_df(category, year, stars)
    d[year] = calc_avg_lexicon_coverage(df_tmp.sample(100000, replace=True), lexicon)
    print("Finished {}".format(year))


def calc_avg_top_coverage(df, lexicon):
    vocab, score_arr = create_vocab(lexicon)
    vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', vocabulary=vocab, binary=False)
    feature_matrix = vectorizer.fit_transform(df['reviews'].values.astype('U')).toarray()
    col_sums = feature_matrix.sum(axis=0)
    col_sums.sort()
    return sum(col_sums[-75:])/sum(col_sums), 0


def calc_top_coverage(year, category, sample_size, lexicon, task_dict, std_dict, stars):
    df_tmp = utils.get_star_by_year_df(category, year, stars)
    task_dict[year], std_dict[year] = calc_avg_top_coverage(df_tmp.sample(sample_size, replace=True), lexicon)
    print("Finished {}".format(year))


def parallel_calculation_per_year(calc_args, calc_func, start_year, end_year, stars=None, do_print=True, desc=None,
                                  max_processes = None):
    file_name = utils.format_file_name(calc_args[0], desc, stars)
    if os.path.isfile(file_name):
        print("{} already exists".format(file_name))
        task_dict, std_dict = utils.load_file(file_name)
        return task_dict, std_dict
    manager = Manager()
    task_dict = manager.dict()
    std_dict = manager.dict()
    if max_processes:
        pool = Pool(processes=max_processes)  # Create a process pool
    else:
        pool = Pool()

    for i in range(start_year, end_year + 1):
        if stars:
            per_year_calc_args = (i,) + calc_args + (task_dict, std_dict, stars)
        else:
            per_year_calc_args = (i,) + calc_args + (task_dict, std_dict)

        pool.apply_async(calc_func, args=per_year_calc_args)

    pool.close()
    pool.join()

    return task_dict, std_dict


def calc_avg_len(df):
    reviews = df['reviews'].tolist()
    length_sum = 0
    count = 0
    lengths = []
    for r in reviews:
        try:
            length_sum += len(r.split())
            lengths.append(len(r.split()))
        except:
            count += 1
    avg_length = length_sum/(len(reviews) - count)
    std = np.std(lengths)

    return avg_length, std


def calc_len(year, category, sample_size, task_dict, std_dict, stars):
    df_tmp = utils.get_star_by_year_df(category, year, stars)
    task_dict[year], std_dict[year] = calc_avg_len(df_tmp.sample(sample_size, replace=True))
    print("Finished {}".format(year))





def calc_sen_full_avg(df):
    reviews = df['reviews'].values.astype('U')
    score_sum = 0
    not_in_use = 0
    scores = []
    analyzer = SentimentIntensityAnalyzer()
    for r in reviews:
        vs = analyzer.polarity_scores(r)
        score = vs['compound']
        if score != 0:

            score_sum += vs['compound']
            scores.append(vs['compound'])
        else:
            not_in_use += 1
    result = score_sum/(len(reviews)-not_in_use)
    std = np.std(scores)
    return result, std


def calc_sen_full(year, category, sample_size, task_dict, std_dict, stars):
    df_tmp = utils.get_star_by_year_df(category, year, stars)
    task_dict[year], std_dict[year] = calc_sen_full_avg(df_tmp.sample(sample_size, replace=True))
    print("Finished {}".format(year))


def calc_one_sided(year, category, sample_size, task_dict, std_dict, stars):
    df_tmp = utils.get_star_by_year_df(category, year, stars)
    task_dict[year], std_dict[year] = calc_one_sided_avg(df_tmp.sample(sample_size, replace=True), stars)
    print("Finished {}".format(year))


def calc_one_sided_avg(df, stars):
    if stars == neg_rating:
        sent = 'neg'
        opposite = 'pos'
    else:
        sent = 'pos'
        opposite = 'neg'
    reviews = df['reviews'].values.astype('U')
    score_sum = 0
    analyzer = SentimentIntensityAnalyzer()
    for r in reviews:
        vs = analyzer.polarity_scores(r)
        if vs[sent] > 0 and vs[opposite] == 0:
            score_sum += 1
    result = score_sum/len(reviews)
    return result, 0


def calc_review_num(year, category, d, stars):
    df_tmp = utils.get_star_by_year_df(category, year, stars)
    d[year] = df_tmp.sample(100000, replace=True).shape[0]
    print("Finished {}".format(year))


def create_vocab(lexicon, use_abs=None):
    vocab = {}
    score_arr = []
    i = 0
    for key in lexicon:
        try:
            if float(lexicon[key]) != 0:
                score_arr.append(float(lexicon[key]))
                if use_abs:
                    score_arr[-1] = abs(score_arr[-1])
                vocab[key] = i
                i += 1
        except:
            pass
    score_arr = np.array(score_arr)
    return vocab, score_arr


def get_reviewers_set(df):
    users_set = set(df["reviewers"].tolist())
    return users_set, None


def calc_users(year, category, task_dict, std_dict, stars):
    df_tmp = utils.get_star_by_year_df(category, year, stars)
    task_dict[year], std_dict[year] = get_reviewers_set(df_tmp)
    print("Finished {}".format(year))


def find_persistent_reviewers(category, stars, start_year, end_year):
    reviewers = []
    calc_args = (category,)
    task_dict, std_dict = parallel_calculation_per_year(calc_args, calc_users, start_year, end_year,
                                                        stars, desc="per", do_print=False)
    for year in task_dict.keys():
        reviewers.append(task_dict[year])
    persistent_reviewers = set.intersection(*reviewers)
    print("{} with {} stars has {} per reviewers from {} up to {}".format(category, stars, len(persistent_reviewers),
                                                                          start_year, end_year))
    return persistent_reviewers




