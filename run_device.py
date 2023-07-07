import utils
import probe
import pickle
import sys

if __name__ == '__main__':

    sample_size = 100000
    categories = ["Booking-pc", "Booking-mobile"]
    start_years = [2018, 2018]
    end_years = [2018, 2018]
    line_styles = ['-', '--', '-.']
    colors = ["blue", "orange", "green"]
    titles = ["Word level sentiment", "Enhanced sentiment", "Absolute sentiment intensity", "#words",
              "% of dichotomous reviews", "% of frequent sentiment words use"]
    display_titles = ["Sentiment", "Enhanced", "Absolute", "#words", "one-sided", "freq"]
    functions = [probe.calc_sen, probe.calc_sen_full, probe.calc_abs, probe.calc_len, probe.calc_one_sided,
                  probe.calc_top_coverage]
    add_lexicon = [True, False, True, False, False, True]
    force_positive = [False, False, False, True, True, True]

    sen = [10, 10]
    lexicon = utils.load_lexicon("data/vader_lexicon.txt")
    max_proc = None
    if len(sys.argv) > 1:
        max_proc = int(sys.argv[1])
    else:
        max_proc = None
    for t in range(len(functions)):

        print(titles[t])

        dicts = []
        std_dicts = []
        for i in range(2):
            if add_lexicon[t]:
                calc_args = (categories[i], sample_size, lexicon)
            else:
                calc_args = (categories[i], sample_size)
            task_dict, std_dict = probe.parallel_calculation_per_year(calc_args, functions[t], start_years[i],
                                                                      end_years[i],
                                                                      sen[i], desc=titles[t], max_processes = max_proc)

            dicts.append(task_dict)
            std_dicts.append(std_dict)
            print("The {} for {} is {}".format(titles[t], categories[i], round(task_dict[start_years[i]],3)))




