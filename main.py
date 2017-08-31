import caching
import dataio
import hand_labeling
import visualization
import sys


if __name__ == '__main__':
    caching.cached_functions[sys.argv[1]](*(sys.argv[2:]))