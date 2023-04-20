import os


def make_dirs(folders):
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as exc:
            if exc.errno != os.errno.EEXIST:
                raise
            pass


def data_in_folder(files, folder):
    out = True
    for file in files:
        if file in os.listdir(folder):
            continue
        else:
            return False
    return out
