import contextlib
import os
import datetime
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


ROOT_DIR = os.getcwd()


_fn_stack = []
_cached_fns = set()


@contextlib.contextmanager
def change_directory(loc):
    loc = '%s/%s' % (ROOT_DIR, loc)
    if not os.path.exists(loc):
        os.makedirs(loc)

    init_dir = os.getcwd()
    os.chdir(loc)
    yield
    os.chdir(init_dir)


def read_input_dir(loc=''):
    return change_directory('input/%s' % loc)


def read_log_dir():
    assert _fn_stack, "Can't read log dir outside of a cached function."
    return change_directory('log/%s' % _fn_stack[-1][1])


def _strargs(*args, **kwargs):
    ret = [repr(x) for x in args]
    ret += sorted(['%s=%s' % (k, repr(v)) for k, v in kwargs.items()])
    return ', '.join(ret)


def _sanitize_dirname(name):
    banned = '~#%&*{}\\:<>?/|".'
    return ''.join(x for x in name if x not in banned) or '_'


class CachedFunction(object):
    def __init__(self, fn, version, static, *deps):
        assert fn.__name__ not in _cached_fns, "Can't have two cached functions with the same name."
        _cached_fns.add(fn.__name__)

        self.version = version
        self.ancestors = set.union({self}, *(x.ancestors for x in deps))
        self.version = sum(x.version for x in self.ancestors)
        self.static = static
        self._fn = fn

    def __call__(self, *args, **kwargs):
        strargs = _strargs(*args, **kwargs)
        dirname = _sanitize_dirname(strargs)
        indent = '| ' * len(_fn_stack)
        called = '%s(%s) v%s' % (self._fn.__name__, strargs, self.version)
        root = 'static' if self.static else 'cache'
        path = '%s/%s/%s' % (self._fn.__name__, self.version, dirname)

        print('%s|-> executing %s ' % (indent, called))
        _fn_stack.append((self, path))
        t0 = time.time()
        with change_directory('%s/%s' % (root, path)):
            ret = self._fn(*args, **kwargs)
        delta = datetime.timedelta(seconds=time.time()-t0)
        _fn_stack.pop()
        print('%s|-> completed %s [%s]' % (indent, called, str(delta)))

        return ret


def cached(*deps, version=0, static=False):
    def decorator(fn):
        return CachedFunction(fn, version, static, *deps)

    return decorator
