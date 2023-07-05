import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib.font_manager as font_manager
from scipy.stats.stats import pearsonr

def load_lexicon(path):
    word = 0
    score = 1
    sentiment_dict = {}
    file = open(path, 'r')
    lines = file.readlines()
    for line in lines:
        line_list = line.split()
        if len(line_list) < 14:
            sentiment_dict[line_list[word]] = line_list[score]
    return sentiment_dict


def load_secondary_lexicon(path):
    word = 0
    score = 1
    sentiment_dict = {}
    file = open(path, 'r')
    lines = file.readlines()
    for line in lines:
        line_list = line.split()
        if len(line_list) < 3:
            sentiment_dict[line_list[word]] = line_list[score]
    return sentiment_dict


def get_lexicons_similarity(path_lex_p, path_lex_s):
    prime_lexicon = load_lexicon(path_lex_p)
    second_lexicon = load_secondary_lexicon(path_lex_s)
    intersection = prime_lexicon.keys() & second_lexicon.keys()
    prime_scores = []
    second_scores = []
    for key in intersection:
        prime_scores.append(float(prime_lexicon[key]))
        second_scores.append(float(second_lexicon[key]))
    print("overall {} keys".format(len(intersection)))
    print(pearsonr(prime_scores, second_scores))


def get_extreme_lexicon(lexicon, threshold, sent):
    extreme_lexicon = {}
    for key in lexicon:
        if sent == "pos":
            if float(lexicon[key]) > threshold:
                extreme_lexicon[key] = lexicon[key]
        elif sent == "neg":
            if float(lexicon[key]) < threshold:
                extreme_lexicon[key] = lexicon[key]
        else:
            print("please enter valid sent")
            return
    print(len(extreme_lexicon))
    return extreme_lexicon


def get_extreme_reviews(extreme_lexicon, reviews):
    extreme_reviews = []
    len_sum = 0
    count = 0
    for rev in reviews:
        if isinstance(rev, str):
            if len(rev) < 300000000:
                words = rev.split(" ")
                for key in extreme_lexicon:
                    if key in words:
                        extreme_reviews.append(rev)
                        #print("###############################")
                        #print(rev)

                        break
    extreme_reviews = sorted(extreme_reviews, key=len)
    for i in range(10):
        print("###########\n" + extreme_reviews[i])
        words = extreme_reviews[i].split(" ")
        for key in extreme_lexicon:
            if key in words:
                print(key)
        count += 1
        len_sum += len(extreme_reviews[i])
        print(len(extreme_reviews[i]))



    print(len_sum/count)





def load_AoA(path):
    AoA_dict = {}
    df = pd.read_csv(path)
    df = df[df['AoArating'].notna()]
    words = df["WORD"].tolist()
    AoA = df["AoArating"].tolist()
    length = len(words)
    for i in range(len(words)):
        AoA_dict[words[i]] = AoA[i]
    return AoA_dict


def slice_data_by_year(df, start_year, end_year, category):
    for i in range(start_year, end_year + 1):
        df_tmp = df[df["years"] == i].copy()
        filename = "data/{}_by_year/{}_df".format(category+"-pc", i)
        with open(filename, 'wb') as f:
            pickle.dump(df_tmp, f)


def get_star_by_year_df(category, year, stars):
    if stars == 1:
        sen = "neg"
    else:
        sen = "pos"
    filename = "data/final_data/{}/{}/{}_df".format(category, sen, year)
    with open(filename, 'rb') as f:
        df_tmp = pickle.load(f)
    return df_tmp





def get_time_series(start, end, task_dict, std_dict=None):
    years, values, std = [], [], []
    for i in range(start, end+1):
        years.append(i)
        values.append(task_dict[i])
        if std_dict:
            std.append(std_dict[i])
    if std_dict:
        return years, values, std
    return years, values


def create_figure(categories, start_years, end_years, dicts, x_name, y_name, line_styles, sen, colors, std_dicts,
                  markers=None):
    hfont = {'fontname': 'Times New Roman', 'size': 'xx-large', 'fontweight': 'bold'}
    plt.locator_params(axis='x', nbins=3)
    for i in range(len(categories)):
        temp_category = categories[i].replace("IMDB", "IMDb")
        temp_category = temp_category.replace("help", "h")
        if std_dicts:
            years, values, std = get_time_series(start_years[i], end_years[i], dicts[i], std_dicts[i])
            for j in range(len(years)):
                years[j] -= start_years[i] + 1
            if markers:
                plt.plot(years, values, line_styles[i], label=temp_category, color=colors[i], marker=markers[i])
            else:
                plt.plot(years, values, line_styles[i], label=temp_category, color=colors[i])
        else:
            years, values = get_time_series(start_years[i], end_years[i], dicts[i])
            plt.plot(years, values, line_styles[i], label=categories[i])
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='bold',
                                       style='normal', size='small')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(categories), prop=font)
    #plt.xlabel(x_name, **hfont)
    plt.ylabel(y_name, **hfont)
    if sen == "negative":
        title = "Negative"
    else:
        title = "Positive"

    plt.title(title, fontname='Times New Roman', size='xx-large', fontweight='bold')
    plt.savefig('figs/{}_{}_pr.png'.format(y_name, sen), dpi=300)
    plt.clf()

def create_figure_main(categories, start_years, end_years, dicts, x_name, y_name, line_styles, sen, colors, std_dicts,
                      markers=None):
    directory_path = "figs/main"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    hfont = {'fontname': 'Times New Roman', 'size': 'xx-large', 'fontweight': 'bold'}
    plt.locator_params(axis='x', nbins=5)
    for i in range(len(categories)):
        temp_category = categories[i].replace("IMDB", "IMDb")
        temp_category = temp_category.replace("help", "h")
        if std_dicts:
            years, values, std = get_time_series(start_years[i], end_years[i], dicts[i], std_dicts[i])
            if markers:
                plt.plot(years, values, line_styles[i], label=temp_category, color=colors[i], marker=markers[i])
            else:
                plt.plot(years, values, line_styles[i], label=temp_category, color=colors[i])
        else:
            print(start_years, end_years, dicts, i)
            years, values = get_time_series(start_years[i], end_years[i], dicts[i])

            plt.plot(years, values, line_styles[i], label=categories[i])
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='bold',
                                       style='normal', size='small')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(categories), prop=font)
    # plt.xlabel(x_name, **hfont)
    plt.ylabel(y_name, **hfont)
    if sen == "negative":
        title = "Negative"
    else:
        title = "Positive"

    plt.title(title, fontname='Times New Roman', size='xx-large', fontweight='bold')
    plt.savefig(directory_path+'/{}_{}.png'.format(y_name, sen), dpi=300)
    plt.clf()


def save_file(file, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        pickle.dump(file, f)


def load_file(filename):
    with open(filename, 'rb') as f:
        file = pickle.load(f)
    return file


def format_file_name(category, desc, stars):
    file_name_desc = desc.replace(' ', '_')
    filename = "results/{}_{}_{}_stars".format(category, file_name_desc, stars)
    return filename


def check_if_file_exists(category, desc, stars):
    file_name_desc = desc.replace(' ', '_')
    filename = "results/{}_{}_{}_stars".format(category, file_name_desc, stars)
    return os.path.isfile(filename)





def prepare_helpful(category, start_year, end_year, sen, minimal_freq=50):
    for i in range(start_year, end_year + 1):
        df = get_star_by_year_df(category, i)
        df = df.fillna('0')
        if category == "Amazon" or category =="IMDB":
            print(category)
            df['votes'] = df['votes'].map(lambda x: x.replace(",", ""))
            df['votes'] = df['votes'].astype(int)

        if category == "IMDB":
            df['overall-votes'] = df['overall-votes'].map(lambda x: x.replace(",", ""))
            df['overall-votes'] = df['overall-votes'].astype(int)
            df['votes'] = df['votes'].astype(int)
            df = df[df['votes']*3 > 2*df["overall-votes"]]

        df_neg = df[df["scores"] == sen[0]].copy()
        df_pos = df[df["scores"] == sen[1]].copy()
        if df_neg.shape[0] < 1000 or df_pos.shape[0] < 1000:
            print("year {} is problematic, category {}".format(i, category))
        df_neg = df_neg[df_neg["votes"] > minimal_freq]
        df_pos = df_pos[df_pos["votes"] > minimal_freq]
        filename = "data/{}_by_year/{}_df".format(category+"-help", i)
        with open(filename, 'wb') as f:
            pickle.dump(pd.concat([df_neg, df_pos], axis=0), f)




