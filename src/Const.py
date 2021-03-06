import sys


class _const:
    class ConstError(TypeError): pass

    class ConstCaseError(ConstError): pass

    def __init__(self):
        pass

    def __setattr__(self, name, value):
        if self.__dict__.has_key(name):
            raise self.ConstError, "Can not change const %s" % name
        if not name.isupper():
            raise self.ConstCaseError, "const name %s is not all upper"  % name
        self.__dict__[name] = value


sys.modules[__name__] = _const()
