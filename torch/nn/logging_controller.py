import sys
from pathlib import Path
import os

class Logger:

    def __init__(self, *filepath):
        self.console = sys.stdout
        for f in filepath:
            (Path(f) / "..").mkdir(parents=True, exist_ok=True)
            gitignore = Path(f) / "../.gitignore"
            if not gitignore.exists():
                gitignore.touch()
                with open(gitignore, "w") as f:
                    f.write("*")
        self.file = [open(f, "a") for f in filepath]

    def write(self, message):
        self.console.write(message)
        for f in self.file:
            if f is not None:
                f.write(message)

    def flush(self):
        self.console.flush()
        for f in self.file:
            if f is not None:
                f.flush()

    def add_logfile(self, *filepath):
        for f in filepath:
            self.file.append(open(f, 'a'))

    def remove_logfile(self, filepath):
        assert type(filepath) == str, "filepath has to be str"
        self.file = [f.close() \
            if f is not None and os.path.abspath(f.name) == os.path.abspath(filepath)\
            else f for f in self.file]

    def change_logfile(self, _out, _in):
        assert type(_in) == str and type(_out) == str, "filepath has to be str"
        self.add_logfile(_in)
        self.remove_logfile(_out)

    def allclose(self):
        for f in self.file:
            if f is not None:
                f.close()
        sys.stdout = self.console

