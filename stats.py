import numpy as np
import pickle
import glob
import pymannkendall as mk
from statsmodels.stats.multitest import fdrcorrection

def dict2list(years_dict):
    sorted_by_year = [years_dict[key] for key in sorted(years_dict.keys(), reverse=False)]
    return sorted_by_year


def calculate_my_idx(years_list):
    z = [ ]
    l = len(years_list)
    for i in range(l-1):
        diff = years_list[i+1] - years_list[i]

        z.append(diff ** 2)
    a = [ ]
    for i in range(l-1):
        a.append(np.sqrt(np.mean(z[0:(i+1)])))
    return np.mean(a)


def print_stats(dir_path):
    file_names = [f for f in glob.glob("{}/*".format(dir_path))]
    p_values = []
    for file_name in file_names:
        with open(file_name, 'rb') as f:
            years_dict = pickle.load(f)[0]
        print("###################")
        print(file_name)
        years_list = dict2list(years_dict)
        idx = calculate_my_idx(years_list)
        #print(idx)
        #print(years_list)
        if "IMDb" in dir_path:
            years_list = years_list[10:]
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(years_list)
        print("trend = {}, H = {} with P-value of {}".format(trend, h, p))
        p_values.append(p)
    return p_values

if __name__ == '__main__':
    p_values = []
    datasets = ["Amazon", "Yelp", "IMDb"]
    for dataset in datasets:
        p_values.extend(print_stats("results/" + dataset))
    _, corrected_pvals, = fdrcorrection(p_values, method = 'negcorr')
    fdr = np.mean(corrected_pvals < 0.05)
    print(_)




