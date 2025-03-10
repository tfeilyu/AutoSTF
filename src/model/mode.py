import enum


class Mode(enum.Enum):
    NONE = 0
    ONE_PATH_FIXED = 1
    ONE_PATH_RANDOM = 2
    TWO_PATHS = 3
    ALL_PATHS = 4


def get_mode(name):
    if name == 'NONE':
        return Mode.NONE
    elif name == 'ONE_PATH_FIXED':
        return Mode.ONE_PATH_FIXED
    elif name == 'ONE_PATH_RANDOM':
        return Mode.ONE_PATH_RANDOM
    elif name == 'TWO_PATHS':
        return Mode.TWO_PATHS
    elif name == 'ALL_PATHS':
        return Mode.ALL_PATHS
    else:
        print('')


def create_mode(mode_name):
    name2mode = {
        'NONE': Mode.NONE,
        'ONE_PATH_FIXED': Mode.ONE_PATH_FIXED,
        'ONE_PATH_RANDOM': Mode.ONE_PATH_RANDOM,
        'TWO_PATHS': Mode.TWO_PATHS,
        'ALL_PATHS': Mode.ALL_PATHS
    }
    mode = name2mode[mode_name]

    return mode