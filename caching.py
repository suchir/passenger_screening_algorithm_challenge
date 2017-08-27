import contextlib
import os
import hashlib


ROOT_DIR = os.getcwd()


@contextlib.contextmanager
def change_directory(dir):
    dir = f'{ROOT_DIR}\\{dir}'
    if not os.path.exists(dir):
        os.makedirs(dir)

    init_dir = os.getcwd()
    os.chdir(dir)
    yield
    os.chdir(init_dir)


def read_input_dir(dir=''):
    return change_directory(f'input\\{dir}')


class CachedFunction(object):
    def __init__(self, fn, version, *deps):
        self.version = version
        self.ancestors = set.union({self}, *(x.ancestors for x in deps))
        self.version = sum(x.version for x in self.ancestors)
        self.fn_name = fn.__name__
        self.dirname = f'{self.fn_name}-{self.version}'
        self.cache = {}
        self._fn = fn

    def __call__(self, mode):
        if mode not in self.cache:
            print(f'running {self.fn_name}-{mode}... ')
            with change_directory(f'cache\\{self.dirname}-{mode}'):
                self.cache[mode] = self._fn(mode)
            print(f'{self.fn_name}-{mode} completed')

        return self.cache[mode]


def cached(*deps, version=0):
    def decorator(fn):
        return CachedFunction(fn, version, *deps)

    return decorator
