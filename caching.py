import contextlib
import os
import hashlib


ROOT_DIR = os.getcwd()


@contextlib.contextmanager
def change_directory(dir):
    dir = '%s\\%s' % (ROOT_DIR, dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    init_dir = os.getcwd()
    os.chdir(dir)
    yield
    os.chdir(init_dir)


def read_input_dir(dir=''):
    return change_directory('input\\%s' % dir)


cached_functions = {}


class CachedFunction(object):
    def __init__(self, fn, version, *deps):
        self.version = version
        self.ancestors = set.union({self}, *(x.ancestors for x in deps))
        self.version = sum(x.version for x in self.ancestors)
        self.dirname = '%s-%s' % (fn.__name__, self.version)
        self._fn = fn
        cached_functions[fn.__name__] = self

    def __call__(self, *args):
        strargs = '-'.join(str(arg) for arg in args)
        print('running %s%s... ' % (self._fn.__name__, args))
        with change_directory('cache\\%s-%s' % (self.dirname, strargs)):
            ret = self._fn(*args)
        print('%s%s completed' % (self._fn.__name__, args))
        return ret


def cached(*deps, version=0):
    def decorator(fn):
        return CachedFunction(fn, version, *deps)

    return decorator
