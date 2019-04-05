import os
import numpy as np

from open_files import *

base_path = '/media/aich/DATA/tmp_models/vgg_cells/results';
res_dir_list = ['N32_1', 'N50_1'];

def gen_file_stats(file_path) :
    fhand = get_file_handle(file_path, 'rb');
    next(fhand); # skip first line

    target_list = [];
    pred_list = [];
    for line in fhand :
        line = line.rstrip(); # cut newline
        path_, target, pred = line.split(',');
        target = int(target);
        pred = int(pred);
        target_list.append(target);
        pred_list.append(pred);

    fhand.close();

    print_file_stats(os.path.basename(file_path), target_list, pred_list);


def print_file_stats(filename_csv, target_list, pred_list) :
    target_list = np.array(target_list);
    pred_list = np.array(pred_list);
    dif_ = pred_list - target_list;
    perc_dif_ = abs(dif_).astype('f2')/target_list; # percentage of deviation w.r.t. total count
    overest = dif_[dif_>0].sum();
    underest = abs(dif_[dif_<0]).sum();
    total_count = target_list.sum();
    MAE_perc = perc_dif_.mean(); # cowc paper
    RMSE_perc = np.sqrt((perc_dif_ * perc_dif_).mean()); # cowc paper
    MAE = abs(dif_).astype('f2').mean(); # LPN paper
    RMSE = np.sqrt( np.power( abs(dif_), 2 ).mean() );

    total_error = float(abs(dif_).sum())/target_list.sum();

    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print '======= Stats for file {} ======='.format(filename_csv);
    print '==== Granular Test Statistics ====';
    print 'Number of test samples = {}'.format(len(target_list));
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
    for res_dir in res_dir_list :
        print "Setup = {}".format(res_dir);
        res_dir = os.path.join(base_path, res_dir);
        for file_ in sorted(os.listdir(res_dir)) :
            if not file_.endswith('.csv') :
                continue;
            gen_file_stats(os.path.join(res_dir, file_));
        print '===============================================';


if __name__ == '__main__' :
    main();
