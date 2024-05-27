class opt2d:
    def __init__(self, function, hx = 0.001, hy = 0.001) -> None:
        self.f = function
        self.hx = hx
        self.hy = hy
    
    