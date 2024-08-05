from sympy import sqrt, Matrix, symbols
from sympy.physics.quantum import TensorProduct
from . import apply_from_kraus
from itertools import product
from su3 import *

p10, p20, p21 = symbols('p10 p20 p21', real=True, positive=True)


T_2 = {
    0: Matrix([[1, 0], [0, sqrt(1 - p10)]]),
    1: Matrix([[0, sqrt(p10)], [0, 0]])
}

T_3 = {
    0: Matrix([[1, 0, 0], [0, sqrt(1 - p10), 0], [0, 0, sqrt(1 - p20 - p21)]]),
    1: sqrt(p10)/2*(l1 + I*l2),
    2: sqrt(p20)/2*(l4 + I*l5),
    3: sqrt(p21)/2*(l6 + I*l7),
}

T_23 = list(TensorProduct(i,j) for i, j in product(T_2.values(), T_3.values()))
T_33 = list(TensorProduct(i,j) for i, j in product(T_3.values(), T_3.values()))


def amplitude_damping(rho, make_real):
    x, _ = rho.shape

    if x == 6:
        Ks = T_23
    elif x == 9:
        Ks = T_33

    result = apply_from_kraus(Ks, rho).subs([(a.conjugate(), a) for a in make_real])
    result = result.applyfunc(lambda x: x.factor())
    
    return result
