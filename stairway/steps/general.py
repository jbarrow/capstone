
import os

def print_data(data, message=""):
    print message ; print data ; return data

def r_load_pairs(directory='', exts=[], master='.txt'):
    files = []
    for item in os.listdir(directory):
        if item[0] == '.': continue
        current = os.path.join(directory, item).strip()
        if os.path.isdir(current): files.extend(r_load_pairs(current, exts, master))
        if current[-len(master):] == master:
            current = current[:-len(master)]
            files.append(tuple([current+ext for ext in exts]))
    return files

def cast(*args):
    return args[-1](*args[:-1])
