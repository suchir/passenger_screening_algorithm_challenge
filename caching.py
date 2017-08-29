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


class CachedFunction(object):
    def __init__(self, fn, version, *deps):
        deps = sorted(deps, key=lambda x: x.hash)
        deps_hash = hashlib.sha256(''.join(x.hash for x in deps).encode()).hexdigest()[:16]
        self.hash = '%s-%s-%s' % (fn.__name__, version, deps_hash)
        self._cache = {}
        self._fn = fn

    def __call__(self, *args):
        if args not in self._cache:
            strargs = '-'.join(str(arg) for arg in args)
            print('running %s%s... ' % (self._fn.__name__, args))
            with change_directory('cache\\%s-%s' % (self.hash, strargs)):
                self._cache[args] = self._fn(*args)
            print('%s%s completed' % (self._fn.__name__, args))

        return self._cache[args]


def cached(*deps, version=0):
    def decorator(fn):
        return CachedFunction(fn, version, *deps)

    return decorator
