import contextlib
import os
import datetime
import time
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


ROOT_DIR = os.getcwd()
REMOTE_ROOT_DIR = '/home/Suchir/passenger_screening_algorithm_challenge'

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


def cache_to_log(cache):
    return 'log/%s' % cache[6:]


def read_log_dir():
    assert _fn_stack, "Can't read log dir outside of a cached function."
    return change_directory(cache_to_log(_fn_stack[-1][1]))


def _strargs(*args, **kwargs):
    ret = [repr(x) for x in args]
    ret += sorted(['%s=%s' % (k, repr(v)) for k, v in kwargs.items()])
    return ', '.join(ret)


def _sanitize_dirname(name):
    banned = '~#%&*{}\\:<>?/|".'
    return ''.join(x for x in name if x not in banned) or '_'


class CachedFunction(object):
    def __init__(self, fn, version, subdir, *deps):
        assert fn.__name__ not in _cached_fns, "Can't have two cached functions with the same name."
        _cached_fns.add(fn.__name__)

        self.version = version
        self.ancestors = set.union({self}, *(x.ancestors for x in deps))
        self.version = sum(x.version for x in self.ancestors)
        self.subdir = subdir
        self._fn = fn

    def _path(self, *args, **kwargs):
        dirname = _sanitize_dirname(_strargs(*args, **kwargs))
        path = '%s/%s/%s' % (self._fn.__name__, self.version, dirname)
        if self.subdir:
            path = '%s/%s' % (self.subdir, path)
        path = 'cache/%s' % path
        return path

    def __call__(self, *args, **kwargs):
        indent = '| ' * len(_fn_stack)
        called = '%s(%s) v%s' % (self._fn.__name__, _strargs(*args, **kwargs), self.version)
        print('%s|-> executing %s ' % (indent, called))
        t0 = time.time()

        path = self._path(*args, **kwargs)
        _fn_stack.append((self, path))
        with change_directory(path):
            ret = self._fn(*args, **kwargs)
        _fn_stack.pop()

        delta = datetime.timedelta(seconds=time.time()-t0)
        print('%s|-> completed %s [%s]' % (indent, called, str(delta)))

        return ret

    def sync_cache(self, box, *args, **kwargs):
        cache_path = self._path(*args, **kwargs)
        log_path = cache_to_log(cache_path)

        for path in (cache_path, log_path):
            if not os.path.exists(path):
                os.makedirs(path)
            remote_path = '%s:%s/%s/*' % (box, REMOTE_ROOT_DIR, path)
            subprocess.check_call(['gcloud', 'compute', 'scp', '--recurse', remote_path, path],
                                  shell=True)


def cached(*deps, version=0, subdir=None):
    def decorator(fn):
        return CachedFunction(fn, version, subdir, *deps)

    return decorator
