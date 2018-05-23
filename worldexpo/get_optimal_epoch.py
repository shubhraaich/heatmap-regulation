""" https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394 """
import numpy as np
import os

from open_files import *

win_size = 1;
val_file_prefix = 'val_diff_old_';
if os.path.isfile(val_file_prefix + "simple") :
    val_file_name = val_file_prefix + "simple";
elif os.path.isfile(val_file_prefix + "gap_gas") :
    val_file_name = val_file_prefix + "gap_gas";
else :
    print "validation difference file does not exist."
    exit();

val = np.array(load_pickle(val_file_name));

cumsum, moving_avg = [0.], []
for i, x in enumerate(val, 1):
    cumsum.append(cumsum[i-1] + x)
    if i < win_size:
        continue;
    ma_ = (cumsum[i] - cumsum[i-win_size])/win_size
    #can do stuff with moving_ave here
    moving_avg.append(ma_);

moving_avg = np.array(moving_avg);
ind_optimal = np.argmin(moving_avg);
print "Validation errors = {}".format(val);
print "Optimal set of errors = {}".format(val[ind_optimal-win_size+1 : ind_optimal+1]);
print "Optimal epoch = {}".format(ind_optimal+1-(win_size-1)/2);
