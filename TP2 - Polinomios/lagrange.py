from polis import poly 
import numpy as np
import matplotlib.pyplot as plt

class lagrange(poly):

    def __init__(self, data):
        '''
        Recibe una lista de tuplas con los puntos a interpolar.
        '''
        self.data = data
        self.x_values = [tupla[0] for tupla in self.data]
        self.y_values = [tupla[1] for tupla in self.data]
        self.lagrange_coefs = self._get_lagrange_coefs()
        self.lagrange_grade = len(self.lagrange_coefs) - 1
        super().__init__(self.lagrange_grade, self.lagrange_coefs)
    
    def _get_lagrange_coefs(self):
        
        lagrange_poly = 0
        for i, yi in enumerate(self.y_values):
            wi = 1 
            xi = self.x_values[i]
            
            for j, xj in enumerate(self.x_values):
                if xi == xj:
                    continue
                numerator = poly(n=1, coefs=[-xj, 1]) # x - xj
                denominator = xi - xj
                monomial = numerator // denominator
                wi *= monomial
                
            lagrange_poly += yi * wi
        
        return lagrange_poly.coefs
    
    
    def poly_plt(self, a, b, extra_points=None, **kwargs):
        new_x_values = np.linspace(a, b, 100)
        new_y_values = [self(x) for x in new_x_values]
        plt.plot(new_x_values, new_y_values, **kwargs)
        if extra_points:
            extra_x_values = [point[0] for point in extra_points]
            extra_y_values = [point[1] for point in extra_points]
            plt.scatter(extra_x_values, extra_y_values)
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.show()

    
    # Graficar los puntos


a = lagrange([(0, 2), (2, 3), (4, 8), (3, 4),(1, 1)])
# a.poly_plot(0, 10, color='red')
print(a)