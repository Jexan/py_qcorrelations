from sympy import sqrt, eye, symbols
from . import apply_from_kraus
from .. import base_su2, base_su3
from itertools import product
from sympy.physics.quantum import TensorProduct


a = symbols('a', real=True, positive=True)

K_2 = {0: sqrt(1 - 3*a/4)*eye(2)}
K_2.update({i: sqrt(a/4)*li for i, li in enumerate(base_su2[1:], 1)})

K_3 = {0: sqrt(1 - 8*a/9)*eye(3)}
K_3.update({i: sqrt(a/6)*li for i, li in enumerate(base_su3[1:], 1)})

K_23 = list(TensorProduct(i,j) for i, j in product(K_2.values(), K_3.values()))
K_33 = list(TensorProduct(i,j) for i, j in product(K_3.values(), repeat=2))

def depolarize(rho, make_real):
    x, _ = rho.shape

    if x == 6:
        Ks = K_23
    elif x == 9:
        Ks = K_33

    result = apply_from_kraus(Ks, rho).subs([(a.conjugate(), a) for a in make_real])
    result = result.applyfunc(lambda x: x.factor())
    
    return result
