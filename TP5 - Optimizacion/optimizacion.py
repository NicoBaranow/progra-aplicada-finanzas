class opt2d:
    def __init__(self, function, hx = 0.001, hy = 0.001) -> None:
        '''
        function: funcion a optimizar. f(x), donde x es una tupla (x, y)
        hx: paso en x
        hy: paso en y
        '''
        self.f = function
        self.hx = hx
        self.hy = hy
    
    def fx(self, x):
        return (self.f(x[0] + self.hx, x[1]) - self.f(x[0] - self.hx, x[1])) / (2 * self.hx)

    def fy(self, x):
        return (self.f(x[0], x[1] + self.hy) - self.f(x[0], x[1] - self.hy)) / (2 * self.hy)

    def fxx(self, x):
        return (self.fx((x[0] + self.hx, x[1])) - self.fx((x[0] - self.hx, x[1]))) / (2 * self.hx)

    def fyy(self, x):
        return (self.fy((x[0], x[1] + self.hy)) - self.fy((x[0], x[1] - self.hy))) / (2 * self.hy)

    def fxy(self, x):
        return (self.fx((x[0], x[1] + self.hy)) - self.fx((x[0], x[1] - self.hy))) / (2 * self.hy)
