import utils
import probe
import pickle
import sys

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
    max_proc = None
    if len(sys.argv) > 1:
        max_proc = int(sys.argv[1])
    else:
        max_proc = None

    indexes = [0, 2]
    for t in range(6):
        print("###### " + titles[t] + " ######")
        for j in range(2):
            dicts = []
            std_dicts = []
            for i in range(3):
                if add_lexicon[t]:
                    calc_args = (categories[i], sample_size, lexicon)
                else:
                    calc_args = (categories[i], sample_size)
                print("Working on {} reviews with {} stars ".format(categories[i], sen[i][j]))
                task_dict, std_dict = probe.parallel_calculation_per_year(calc_args, functions[t], start_years[i],
                                                                          end_years[i],
                                                                          sen[i][j], desc=titles[t], max_processes = max_proc)
                dicts.append(task_dict)
                std_dicts.append(std_dict)
            if j == 0:
                binary_sen = "negative"
            else:
                binary_sen = "positive"
            utils.create_figure_main(categories, start_years, end_years, dicts, "Years", titles[t], line_styles, binary_sen
                                , colors, std_dicts, markers)
            exit()
