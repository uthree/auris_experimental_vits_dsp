from module.utils.hparams import HParams

def a(a, b):
    print(a, b)

h = HParams(**{"a": 1, "b": 2})
a(**h)
