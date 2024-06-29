from optim import opt2d, polytope

def restriccion(x,y):
    return x**2+y**2+1 #circulo de radio 1

def minima_distancia():
    optimizer = opt2d(restriccion)
    optimizer.contours()

    pltp = polytope([(1,0),(0,1),(-1, 0), (0, -1)],restriccion)
    pt = pltp.polyt_prog()

    optimizer.campo_gradiente(-5, 5, -5, 5, 20, 20, optimizer.gdescent(1, 1, returnAsPoints=True))

    pt = optimizer.gdescent_global(x0=3, y0=0)
    print("Gradiente de f : ", optimizer.grad_call(pt[0], pt[1]))
    print(optimizer.minimumState(optimizer.hessienna(pt[0], pt[1])))


minima_distancia()