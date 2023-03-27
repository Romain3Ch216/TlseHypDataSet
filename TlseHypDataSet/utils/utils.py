import os


def make_dirs(folders):
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as exc:
            if exc.errno != os.errno.EEXIST:
                raise
            pass