import func

class MLP_LAYER:
    def __init__(self, units = 0, in_size = 0, activ_func = func.sigmoid):
        self.units = units
        self.in_size = in_size
        self.activ_func = activ_func

