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


_fn_stack = []


class CachedFunction(object):
    def __init__(self, fn, version, *deps):
        self.version = version
        self.ancestors = set.union({self}, *(x.ancestors for x in deps))
        self.version = sum(x.version for x in self.ancestors)
        self.dirname = '%s-%s' % (fn.__name__, self.version)
        self._fn = fn

    def __call__(self, *args):
        strargs = '-'.join(str(arg) for arg in args)
        indent = '| ' * len(_fn_stack)
        pretty_args = str(args)
        if len(args) == 1:
            pretty_args = pretty_args.replace(',', '')
        print('%s|-> executing %s%s ' % (indent, self._fn.__name__, pretty_args))

        _fn_stack.append(self)
        with change_directory('cache\\%s-%s' % (self.dirname, strargs)):
            ret = self._fn(*args)
        _fn_stack.pop()

        print('%s|-> completed %s%s' % (indent, self._fn.__name__, pretty_args))
        return ret


def cached(*deps, version=0):
    def decorator(fn):
        return CachedFunction(fn, version, *deps)

    return decorator
