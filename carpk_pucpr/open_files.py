import pickle

def get_file_handle(path_, r_w_a) :
    try :
        fhand = open(path_, r_w_a);
    except :
        print("Cannot open file {}".format(path_));
        exit();
    return fhand;

def save_pickle(path_, data) :
    fhand = get_file_handle(path_, 'wb+');
    pickle.dump(data, fhand);
    fhand.close();

def load_pickle(path_) :
    fhand = get_file_handle(path_, 'rb');
    data = pickle.load(fhand);
    fhand.close();
    return data;
