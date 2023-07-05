import utils
import probe
import pickle


if __name__ == '__main__':
    sample_size = 100000
    categories = ["Amazon", "Yelp", "IMDB"]
    start_years = [2000, 2007, 2000]
    end_years = [2018, 2020, 2020]
    line_styles = ['-', '-', '-']
    markers = ['o', 's', '^']
    colors = ["blue", "orange", "green"]
    titles = ["Word level sentiment", "Enhanced sentiment", "Absolute sentiment intensity", "#words",
              "% of dichotomous reviews", "% of frequent sentiment words use"]
    display_titles = ["Sentiment", "Enhanced", "Absolute", "#words", "one-sided", "freq"]
    functions = [probe.calc_sen, probe.calc_sen_full, probe.calc_abs, probe.calc_len, probe.calc_one_sided, probe.calc_top_coverage]
    add_lexicon = [True, False, True, False, False, False, True]
    force_positive = [False, False, False, True, True, True]
    sen = [[1, 5], [1, 5], [1, 10]]
    lexicon = utils.load_lexicon("data/vader_lexicon.txt")



    indexes = [0, 2]
    for t in range(6):
        print(titles[t])
        for j in range(2):
            dicts = []
            std_dicts = []
            for i in range(3):
                if add_lexicon[t]:
                    calc_args = (categories[i], sample_size, lexicon)
                else:
                    calc_args = (categories[i], sample_size)
                task_dict, std_dict = probe.parallel_calculation_per_year(calc_args, functions[t], start_years[i],
                                                                          end_years[i],
                                                                          sen[i][j], desc=titles[t])
                print(task_dict)
                dicts.append(task_dict)
                std_dicts.append(std_dict)
            if j == 0:
                binary_sen = "negative"
            else:
                binary_sen = "positive"
            utils.create_figure(categories, start_years, end_years, dicts, "Years", titles[t], line_styles, binary_sen
                                , colors, std_dicts, markers)

    #utils.create_heat_map(titles, categories, sen, display_titles)
    exit()

    sample_size = 100000
    categories = ["Amazon", "Amazon-pr", "Yelp", "Yelp-pr", "IMDB", "IMDB-pr"]
    output_cateories = ["Amazon-pr", "Yelp-pr", "IMDB-pr"]
    start_years = [2012, 2012, 2013, 2013, 2016, 2016]
    end_years = [2014, 2014, 2015, 2015, 2018, 2018]
    line_styles = ['-', '--', '-', '--', '-', '--']
    markers = ['o', 'o', 's', 's', '^', '^']
    colors = ["blue", "blue", "orange", "orange", "green", "green"]
    titles = ["Word level sentiment", "Enhanced sentiment", "Absolute sentiment intensity", "#words",
              "% of dichotomous reviews", "MTLD", "% of frequent sentiment words use"]
    display_titles = ["Sentiment", "Enhanced", "Absolute", "#words", "one-sided", "MTLD", "freq"]
    functions = [probe.calc_sen, probe.calc_sen_full, probe.calc_abs, probe.calc_len, probe.calc_one_sided,
                 probe.calc_lexical_richness, probe.calc_top_coverage]
    add_lexicon = [True, False, True, False, False, False, True]
    force_positive = [False, False, False, True, True, True, True]
    sen = [[1, 5], [1, 5], [1, 5], [1, 5], [1, 10], [1, 10]]
    AoA = utils.load_AoA("data/AoA.csv")
    lexicon = utils.load_lexicon("data/vader_lexicon.txt")
    intersection = AoA.keys() & lexicon.keys()
    both = {your_key: AoA[your_key] for your_key in intersection}
    for t in range(7):
        print(titles[t])
        for j in range(2):
            dicts = []
            std_dicts = []
            for i in range(6):

                if add_lexicon[t]:
                    calc_args = (categories[i], sample_size, lexicon)
                else:
                    calc_args = (categories[i], sample_size)
                task_dict, std_dict = probe.parallel_calculation_per_year(calc_args, functions[t], start_years[i],
                                                                          end_years[i],
                                                                          sen[i][j], desc=titles[t])
                print(task_dict)
                dicts.append(task_dict)
                std_dicts.append(std_dict)
            if j == 0:
                binary_sen = "negative"
            else:
                binary_sen = "positive"
            utils.create_figure(categories, start_years, end_years, dicts, "Year", titles[t],
                                line_styles, binary_sen
                                , colors, std_dicts, markers)

    exit()



    sample_size = 100000
    categories = ["Amazon", "Amazon-help", "Yelp", "Yelp-help", "IMDB", "IMDB-help"]
    output_cateories = ["Amazon-help", "Yelp-help", "IMDB-help"]
    start_years = [2000, 2000, 2007, 2007, 2000, 2000]
    end_years = [2018, 2018, 2020, 2020, 2020, 2020]
    line_styles = ['-', '--', '-', '--', '-', '--']
    markers = ['o', 'o', 's', 's', '^', '^']
    colors = ["blue", "blue", "orange", "orange", "green", "green"]
    titles = ["Word level sentiment", "Enhanced sentiment", "Absolute sentiment intensity", "#words",
              "% of dichotomous reviews", "MTLD", "% of frequent sentiment words use"]
    display_titles = ["Sentiment", "Enhanced", "Absolute", "#words", "one-sided", "MTLD", "freq"]
    functions = [probe.calc_sen, probe.calc_sen_full, probe.calc_abs, probe.calc_len, probe.calc_one_sided,
                 probe.calc_lexical_richness, probe.calc_top_coverage]
    add_lexicon = [True, False, True, False, False, False, True]
    force_positive = [False, False, False, True, True, True, True]

    sen = [[1, 5], [1, 5], [1, 5], [1, 5], [1, 10], [1, 10]]
    AoA = utils.load_AoA("data/AoA.csv")
    lexicon = utils.load_lexicon("data/vader_lexicon.txt")
    intersection = AoA.keys() & lexicon.keys()
    both = {your_key: AoA[your_key] for your_key in intersection}
    for t in range(7):
        print(titles[t])
        for j in range(2):
            dicts = []
            std_dicts = []
            for i in range(6):
                if add_lexicon[t]:
                    calc_args = (categories[i], sample_size, lexicon)
                else:
                    calc_args = (categories[i], sample_size)
                task_dict, std_dict = probe.parallel_calculation_per_year(calc_args, functions[t], start_years[i],
                                                                          end_years[i],
                                                                          sen[i][j], desc=titles[t])
                print(task_dict)
                dicts.append(task_dict)
                std_dicts.append(std_dict)
            if j == 0:
                binary_sen = "negative"
            else:
                binary_sen = "positive"
            utils.create_figure(categories, start_years, end_years, dicts, "Year", titles[t],
                                line_styles, binary_sen
                                , colors, std_dicts)
    exit()

    sample_size = 100000
    categories = ["Booking-pc", "Booking-mobile"]
    start_years = [2018, 2018]
    end_years = [2018, 2018]
    line_styles = ['-', '--', '-.']
    colors = ["blue", "orange", "green"]
    titles = ["Word level sentiment", "Enhanced sentiment", "Absolute sentiment intensity", "#words",
              "% of dichotomous reviews", "MTLD", "% of frequent sentiment words use"]
    display_titles = ["Sentiment", "Enhanced", "Absolute", "#words", "one-sided", "MTLD", "freq"]
    functions = [probe.calc_sen, probe.calc_sen_full, probe.calc_abs, probe.calc_len, probe.calc_one_sided,
                 probe.calc_lexical_richness, probe.calc_top_coverage]
    add_lexicon = [True, False, True, False, False, False, True]
    force_positive = [False, False, False, True, True, True, True]

    sen = [10, 10, 10]
    AoA = utils.load_AoA("data/AoA.csv")
    lexicon = utils.load_lexicon("data/vader_lexicon.txt")
    intersection = AoA.keys() & lexicon.keys()
    both = {your_key: AoA[your_key] for your_key in intersection}
    for t in range(len(functions)):

        print(titles[t])
        for j in range(1, 2):

            dicts = []
            std_dicts = []
            for i in range(2):
                if add_lexicon[t]:
                    calc_args = (categories[i], sample_size, lexicon)
                else:
                    calc_args = (categories[i], sample_size)
                task_dict, std_dict = probe.parallel_calculation_per_year(calc_args, functions[t], start_years[i],
                                                                          end_years[i],
                                                                          sen[i], desc=titles[t])
                print("{} for {} in {}".format(task_dict[2018], categories[i], titles[t]))
                dicts.append(task_dict)
                std_dicts.append(std_dict)
            if j == 0:
                binary_sen = "negative"
            else:
                binary_sen = "positive"
            utils.create_figure(categories, start_years, end_years, dicts, "Years", titles[t], line_styles, binary_sen
                                , colors, std_dicts)
    exit()

    category="IMDB"
    start_year = 2000
    end_year = 2020
    sen = [1, 10]
    utils.prepare_helpful(category, start_year, end_year, sen, minimal_freq=10)


    categories = ["Amazon", "Yelp", "IMDB"]
    for i in range(len(categories)):
        filename = "data/{}_clean_df".format(categories[i])
        with open(filename, 'rb') as f:
            df_tmp = pickle.load(f)
        if categories[i] == "Yelp" or categories[i] == "IMDB":
            df_tmp = df_tmp[df_tmp['years'] != 2000]
        years = df_tmp['years'].tolist()
        probe.get_histogram(years, "years", "#reviews", "reviews per year", categories[i])




    start_year = 2015
    end_year = 2017
    category = "Booking"
    bdp = BookingDataProvider(category, "data/Booking.csv")
    bdp.construct_dataset()

    filename = filename = "data/{}_clean_df".format(category)
    with open(filename, 'rb') as f:
        df_tmp = pickle.load(f)
    df_tmp = df_tmp[df_tmp["devices"] == "pc"]
    utils.slice_data_by_year(df_tmp, start_year, end_year, category)
    exit()


    sample_size = 100000
    categories = ["Booking-pc", "Booking-mobile"]
    start_years = [2018, 2018]
    end_years = [2018, 2018]
    line_styles = ['-', '--', '-.']
    colors = ["blue", "orange", "green"]
    titles = ["Word level sentiment", "Enhanced sentiment", "Absolute sentiment intensity", "#words",
              "% of dichotomous reviews", "MTLD", "% of frequent sentiment words use"]
    display_titles = ["Sentiment", "Enhanced", "Absolute", "#words", "one-sided", "MTLD", "freq"]
    functions = [probe.calc_sen, probe.calc_sen_full, probe.calc_abs, probe.calc_len, probe.calc_one_sided,
                 probe.calc_lexical_richness, probe.calc_top_coverage]
    add_lexicon = [True, False, True, False, False, False, True]
    force_positive = [False, False, False, True, True, True, True]

    sen = [10, 10, 10]
    AoA = utils.load_AoA("data/AoA.csv")
    lexicon = utils.load_lexicon("data/vader_lexicon.txt")
    intersection = AoA.keys() & lexicon.keys()
    both = {your_key: AoA[your_key] for your_key in intersection}
    for t in range(len(functions)):

        print(titles[t])
        for j in range(1, 2):

            dicts = []
            std_dicts = []
            for i in range(2):
                if add_lexicon[t]:
                    calc_args = (categories[i], sample_size, lexicon)
                else:
                    calc_args = (categories[i], sample_size)
                task_dict, std_dict = probe.parallel_calculation_per_year(calc_args, functions[t], start_years[i],
                                                                          end_years[i],
                                                                          sen[i], desc=titles[t])
                print(task_dict)
                dicts.append(task_dict)
                std_dicts.append(std_dict)

            if j == 0:
                binary_sen = "negative"
            else:
                binary_sen = "positive"
            utils.create_figure(categories, start_years, end_years, dicts, "Years", titles[t], line_styles, binary_sen
                                , colors, std_dicts, force_positive[t])



    exit()







    users = probe.find_persistent_reviewers("Yelp", 1, 2018, 2020)
    print(len(users))
    sen = [[1, 5], [1, 5], [1, 10]]
    categories = ["Amazon", "Yelp", "IMDB"]
    output_cateories = ["Amazon-pr", "Yelp-pr", "IMDB-pr"]
    start_years = [2011, 2013, 2016]
    end_years = [2015, 2015, 2018]
    for i in range(3):
        users_list = []
        for j in range(2):
            users = probe.find_persistent_reviewers(categories[i], sen[i][j], start_years[i], end_years[i])
            users_list.append(users)
        all_users = users_list[0] | users_list[1]

        filename = filename = "data/{}_clean_df".format(categories[i])
        with open(filename, 'rb') as f:
            df_tmp = pickle.load(f)

        utils.slice_data_by_year(df_tmp[df_tmp['reviewers'].isin(all_users)], start_years[i], end_years[i],
                                 categories[i])


    categories = ["Amazon", "Yelp", "IMDB"]
    start_years = [2000, 2007, 2000]
    end_years = [2018, 2021, 2021]
    for i in range(3):
        filename = filename = "data/{}_clean_df".format(categories[i])
        with open(filename, 'rb') as f:
            df_tmp = pickle.load(f)
        utils.slice_data_by_year(df_tmp, start_years[i], end_years[i], categories[i])
    exit()

    sample_size = 100000
    categories = ["Amazon", "Yelp", "IMDB"]
    start_years = [2000, 2007, 2000]
    end_years = [2018, 2020, 2020]
    line_styles = ['-', '-', '-']
    markers = ['o', 's', '^']
    colors = ["blue", "orange", "green"]
    titles = ["Word level sentiment", "Enhanced sentiment", "Absolute sentiment intensity", "#words",
              "% of dichotomous reviews", "MTLD", "% of frequent sentiment words use"]
    display_titles = ["Sentiment", "Enhanced", "Absolute", "#words", "one-sided", "MTLD", "freq"]
    functions = [probe.calc_sen, probe.calc_sen_full, probe.calc_abs, probe.calc_len, probe.calc_one_sided,
                 probe.calc_lexical_richness, probe.calc_top_coverage]
    add_lexicon = [True, False, True, False, False, False, True]
    force_positive = [False, False, False, True, True, True, True]
    sen = [[1, 5], [1, 5], [1, 10]]
    AoA = utils.load_AoA("data/AoA.csv")
    lexicon = utils.load_lexicon("data/vader_lexicon.txt")
    intersection = AoA.keys() & lexicon.keys()
    both = {your_key: AoA[your_key] for your_key in intersection}
    for t in range(len(titles)):
        print(titles[t])
        for j in range(2):
            dicts = []
            std_dicts = []
            for i in range(3):
                if add_lexicon[t]:
                    calc_args = (categories[i], sample_size, lexicon)
                else:
                    calc_args = (categories[i], sample_size)
                task_dict, std_dict = probe.parallel_calculation_per_year(calc_args, functions[t], start_years[i], end_years[i],
                                                                sen[i][j], desc=titles[t])
                print(task_dict)
                dicts.append(task_dict)
                std_dicts.append(std_dict)
            if j == 0:
                binary_sen = "negative"
            else:
                binary_sen = "positive"
            utils.create_figure(categories, start_years, end_years, dicts, "Years", titles[t], line_styles, binary_sen
                                , colors, std_dicts, markers)
    utils.create_heat_map(titles, categories, sen, display_titles)


















    exit()
    category = "IMDB"
    calc_args = (category, )

    year_dict = probe.parallel_calculation_per_year(calc_args, probe.calc_review_num, 2000,
                                                   2021, 10)

    exit()
    sen_dict = probe.parallel_calculation_per_year(calc_args, probe.calc_one_sided, 2007,
                                                   2021, 5)


    exit()

    lexicon = utils.load_lexicon("data/vader_lexicon.txt")
    vocab, _ = probe.create_vocab(lexicon)
    calc_args = (category, 1000, 3)
    sample_dict = probe.parallel_calculation_per_year(calc_args, train.sample_n_samples, 2010,
                                                      2021, 1, False)
    calc_args = (category, 1000, 3, sample_dict, 2021, vocab)
    sample_dict = probe.parallel_calculation_per_year(calc_args, train.train_domain_classifier, 2010,
                                                      2021, 1, True)









