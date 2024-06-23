    def root_find_bisection(self, x, z, a=-10, b=10, tolerance=1e-5, max_iter=100000):
        '''
        Dado un intervalo a, b, encuentra la raíz de f en ese intervalo utilizando el método de la bisección.
        '''
        iter_count = 0
        while iter_count < max_iter:
            root = (a + b) / 2
            f_root = self.f((x, root))
            if abs(f_root - z) <= tolerance: return root
            
            if self.f((x, a)) * f_root < 0: b = root
            else: a = root
            iter_count += 1
        
        print(f"El método de bisección no convergió después de {max_iter} iteraciones.")
        return None

    def contour1(self, x0 = 2, y0 = 2, x_range = (-4,4), tolerance = 1e-5):
        '''
        Grafica las curvas de nivel de f que pasa por el punto x0, y0 
        x_range: tupla con los valores de x inicial y final
        '''
        z = self.f((x0, y0))  # Calculamos el valor de z en el punto (x0, y0)
        x_points = my_linspace(x_range[0], x_range[1],100)  # Generamos los puntos de x
        y_points = []

        for x in x_points:
            # Establecemos un intervalo amplio alrededor de y0 para la búsqueda
            y = self.root_find_bisection(x, z)
            y_points.append(y)
    
        print(x_points, y_points)

    def contour2(self, xparam = 2, yparam = 2, x_range = (-4,4), repetitions = 1000000, alpha = 0.01):
        '''
        Devuelve las coordenadas necesarias para graficar las curvas de nivel de f que pasan por el punto (xparam, yparam)
        '''
        x0 = xparam

        grad = self.gradf((xparam, yparam))
        grad = matrix([grad[0],grad[1]], 2, 1) #convertimos el vector en una matriz de 2x1
        
        r1 = matrix([0, -1, 1, 0],2,2) #matriz de rotacin antihoraria de 90 grados
        r2 = matrix([0, 1, -1, 0],2,2) #matriz de rotacin horarioa de 90 grados

        v1 = r1 * grad #vector perpendicular al gradiente
        v2 = r2 * grad #vector perpendicular al gradiente
        
        xy_list1 = []
        xy_list2 = []

        for _ in range(repetitions):
            x1 = x0 + alpha*v1
            xy_list1.append(x1.elems)

            x0 = x1
            grad = self.gradf((x0.elems[0], x0.elems[1]))
            grad = matrix([grad[0],grad[1]], 2, 1)

            v1 = r1 * grad

        grad = self.gradf((xparam, yparam))
        grad = matrix([grad[0],grad[1]], 2, 1) 
        x0 = xparam

        for _ in range(repetitions):
            x1 = x0 + alpha*v2
            xy_list2.append(x1.elems)

            x0 = x1
            grad = self.gradf((x0.elems[0], x0.elems[1]))
            grad = matrix([grad[0],grad[1]], 2, 1)

            v2 = r2 * grad

        # graficar sentido antihorario
        plt.figure(figsize=(8, 8))
        x_coords1 = [x[0] for x in xy_list1]
        y_coords1 = [y[1] for y in xy_list1]
        plt.plot(x_coords1, y_coords1, label='Rotación antihoraria')

        # graficar sentido horario
        x_coords2 = [x[0] for x in xy_list2]
        y_coords2 = [y[1] for y in xy_list2]
        plt.plot(x_coords2, y_coords2, label='Rotación horaria')

        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.title('Curvas de nivel y campo gradiente')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(x_range)
        plt.ylim(x_range)
        
        plt.show()