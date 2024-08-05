from sympy import Matrix, I, sqrt, symbols, linear_eq_to_matrix
from . import tensor, conmutador, anticonmutador, generar_latex_tensores


l0 = Matrix((
    (1,0,0), 
    (0,1,0), 
    (0,0,1)
))
l1 = Matrix((
    (0,1,0), 
    (1,0,0), 
    (0, 0, 0)
))
l2 = Matrix((
    (0,-I,0), 
    (I,0,0), 
    (0, 0, 0)
))
l3 = Matrix((
    (1,0,0), 
    (0,-1,0), 
    (0, 0, 0)
))
l4 = Matrix((
    (0,0,1), 
    (0,0,0), 
    (1,0,0)
))
l5 = Matrix((
    (0,0,-I), 
    (0,0,0), 
    (I, 0, 0)
))
l6 = Matrix((
    (0,0,0), 
    (0,0,1), 
    (0,1,0)
))
l7 = Matrix((
    (0,0,0), 
    (0,0,-I), 
    (0, I, 0)
))
l8 = 1/sqrt(3)*Matrix((
    (1,0,0), 
    (0,1,0), 
    (0,0,-2)
))

base_su3 = (l0, l1, l2, l3, l4, l5, l6, l7, l8)


def resolucion_base_su_3(A):
    ## RESOLUCIÓN DE COEFICIENTES DE MATRICES EN LA BASE DE SU(3).
    # Se definen 9 variables, que son los 9 coeficientes de las bases
    x0, x1, x2, x3, x4, x5, x6, x7, x8 = symbols('x0 x1 x2 x3 x4 x5 x6 x7 x8')
    syms = (x0, x1, x2, x3, x4, x5, x6, x7, x8)

    # Se genera un sistema de ecuaciones en la matrix
    matriz_de_ecuaciones = l0*x0 + l1*x1 + l2*x2 + l3*x3 + l4*x4 + l5*x5 + l6*x6 + l7*x7 + l8*x8

    M = linear_eq_to_matrix(list(iter(matriz_de_ecuaciones)), *syms)[0]
    M_inv = M.inv()
    return M_inv*A.reshape(9,1)


# CÁLCULO DE LOS TENSORES d, f
def calcular_d_f():
    # TENSORES DESEADOS
    f = tensor()
    d = tensor()

    for indice_1, l_a in enumerate(base_su3):
        for indice_2, l_b in enumerate(base_su3[indice_1:], indice_1):
            res_conm = conmutador(l_a, l_b)
            res_anti_conm = anticonmutador(l_a, l_b)

            indices_base_conm = resolucion_base_su_3(res_conm, M_inv)
            indices_base_anti_conm = resolucion_base_su_3(res_anti_conm, M_inv)

            for i, item in enumerate(iter(indices_base_conm)):
                f[indice_1][indice_2][i] = item/2*I
                f[indice_2][indice_1][i] = -item/2*I

            for i, item in enumerate(iter(indices_base_anti_conm)):
                d[indice_1][indice_2][i] = item/2
                d[indice_2][indice_1][i] = item/2

    return d, f


# Funcion para crear 9 matrices a partir de los tensores de 3 dimensiones
def aplastar_1_nivel(tensor):
    matrices = []
    for i, level_1 in tensor.items():
        matriz = [
            [item for item in level_1[j].values()] for j in level_1
        ]
        matrices.append(Matrix(matriz))    

    return matrices


def clausura(ops):
    return sum((o.adjoint()*o for o in ops), Matrix.zeros(3))


# Comandos para imprimir los tensores f, d como 9 matrices en Latez
def imprimir_tensores_latex(d, f):
    tensores_f = aplastar_1_nivel(f)
    tensores_d = aplastar_1_nivel(d)
    print(generar_latex_tensores(tensores_f, 'f'), generar_latex_tensores(tensores_d, 'd'))


# Si se corre directamente, se imprime un Latex con f, d
if __name__ == '__main__':
    #imprimir_tensores_latex()
    d, f = calcular_d_f()
    import dill

    with open('./tensor.pickle', 'wb') as outf:
        outf.write(dill.dumps(d))


