import importlib.util
ruta_polis = 'progra-aplicada-finanzas\TP2-Polinomios\polis.py'
spec = importlib.util.spec_from_file_location("polis", ruta_polis)
polis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(polis)

a = polis.poly(2,[2,41,51])
print(a.get_expression())