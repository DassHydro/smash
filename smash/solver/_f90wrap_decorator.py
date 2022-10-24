import functools


def character_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        return value.decode().strip()

    return wrapper
