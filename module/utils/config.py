import json


class Config(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = Config(**v)
            self[k] = v

    def __len__(self):
        return len(self.__dict__)

    def __getattr__(*args):
        val = dict.get(*args)
        if isinstance(val, dict):
            return Config(val)
        return val
    
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_json_file(path):
    return Config(**json.load(open(path, encoding='utf-8')))
