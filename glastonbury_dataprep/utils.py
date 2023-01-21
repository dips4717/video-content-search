import pickle
import os 
import errno


def pickle_load(fn):
    with open(fn,'rb') as f:
        data = pickle.load(f)
    print(f'Obj loaded from {fn}')
    return data

def pickle_save(data,fn):
    with open(fn, 'wb') as f:
        data = pickle.dump(data,f)
    print(f'Obj saved to {fn}')

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise