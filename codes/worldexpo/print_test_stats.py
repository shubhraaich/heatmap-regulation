import os
import numpy as np
import pandas as pd

from open_files import *

base_path = '../../data/cc_sjtu';
res_path_list = ['results_simple', 'results_gap_gas'];


def gen_file_stats(res_file) :
    data = pd.read_csv(res_file, header=None);
    target_array = data[1].as_matrix();
    pred_array = data[2].as_matrix();

    file_name = os.path.join(os.path.basename(os.path.dirname(res_file)), os.path.basename(res_file));
    print_file_stats(file_name, target_array, pred_array);


def print_file_stats(filename_csv, target_array, pred_array) :
    dif_ = pred_array - target_array;
    perc_dif_ = abs(dif_).astype('f2')/target_array; # percentage of deviation w.r.t. total count
    overest = dif_[dif_>0].sum();
    underest = abs(dif_[dif_<0]).sum();
    total_count = target_array.sum();
    MAE_perc = perc_dif_.mean(); # cowc paper
    RMSE_perc = np.sqrt((perc_dif_ * perc_dif_).mean()); # cowc paper
    MAE = abs(dif_).astype('f2').mean(); # LPN paper
    RMSE = np.sqrt( np.power( abs(dif_), 2 ).mean() );

    total_error = float(abs(dif_).sum())/target_array.sum();

    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print '======= Stats for file {} ======='.format(filename_csv);
    print '==== Granular Test Statistics ====';
    print 'Number of test samples = {}'.format(len(target_array));
    print 'Total test count = {}'.format(total_count);
    print 'Mean absolute difference = {}'.format(abs(dif_).mean());
    print 'Std absolute difference = {}'.format(abs(dif_).std());
    print 'Overestimate (%) = {} %'.format(float(overest*100)/total_count);
    print 'Underestimate (%) = {} %'.format(float(underest*100)/total_count);
    print 'Deviation (%) = {} %'.format(float((overest + underest)*100)/total_count);
    print 'MAE(perc) = {} %'.format(MAE_perc);
    print 'RMSE(perc) = {} %'.format(RMSE_perc);
    print 'MAE(absolute) = {}'.format(MAE);
    print 'RMSE(absolute) = {}'.format(RMSE);
    print '-----------------';


def main() :
    for res_path in res_path_list :
        tmp_res_path = os.path.join(base_path, res_path);
        for fname in sorted(os.listdir(tmp_res_path)) :
            gen_file_stats(res_file=os.path.join(tmp_res_path, fname));


if __name__ == '__main__' :
    main();
