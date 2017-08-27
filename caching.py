import contextlib
import os
import hashlib


@contextlib.contextmanager
def change_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    init = os.getcwd()
    os.chdir(dir)
    yield
    os.chdir(init)


def read_input_dir(dir=''):
    return change_directory(f'input/{dir}')


class CachedFunction(object):
    def __init__(self, fn, version, *deps):
        self.version = version
        self.ancestors = set.union({self}, *(x.ancestors for x in deps))
        self.version = sum(x.version for x in self.ancestors)
        self.dirname = f'{fn.__name__}-{self.version}'
        self._fn = fn

    def __call__(self, mode):
        with change_directory(f'cache/{self.dirname}-{mode}'):
            return self._fn(mode)


def cached(*deps, version=0):
    def decorator(fn):
        return CachedFunction(fn, version, *deps)

    return decorator
