import os
from datetime import datetime as dt


class Logger:

    def __init__(self, dirpath=None, filename=None):
        self.datetimeformat = "%Y%m%d_%H%M%S"
        self.dt = dt

        if dirpath is not None:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            self.dirpath = dirpath
        else:
            self.dirpath = os.getcwd()

        self.PID = "PID" + str(os.getpid())

        self.datetimeval = self.dt.now().strftime(self.datetimeformat)  # timestamp this logger.

        if filename is not None:
            self.filename = filename
        else:
            self.filename = "summary_" + self.PID + "_" + self.datetimeval + ".txt"

        self._set_file_path()

    def _set_file_path(self):
        self.file_path = os.path.join(self.dirpath, self.filename)

    def write_to_file(self, msg, toprint=True):
        # append message in a new line
        if toprint:
            print(msg)
        outmsg = self.dt.now().strftime(self.datetimeformat) + ": " + msg + "\n"
        with open(self.file_path, "a") as myfile:
            myfile.write(outmsg)

    def change_dirpath(self, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.dirpath = dirpath
        self._set_file_path()

    def change_filename(self, new_fn):
        self.filename = new_fn
        self._set_file_path()

    def clear_all_content(self):
        # see CasualCode3's reply in https://stackoverflow.com/questions/2769061/how-to-erase-the-file-contents-of-text-file-in-python
        with open(self.file_path, "w"):
            pass
