from lib.su2 import base_su2
from lib.su3 import l1, l2, l3, l4, l5, l6, l7, l8
from lib import base_qutrit, density

from sympy import sqrt, Matrix, symbols, eye, S, I
from sympy.physics.quantum import TensorProduct
from . import apply_from_kraus
from .. import base_su3
from itertools import product


a = symbols('a', real=True, positive=True)
p01, p02, p12 = symbols('p01 p02 p12', real=True, positive=True)


# BF
B_2 = {
    0: sqrt(1-p01)*eye(2),
    1: sqrt(p01)*base_su2[1]
}

B_12_3 = {
    1: sqrt(1 - p01)*eye(3),
    2: sqrt(p01)*(base_qutrit[0]* base_qutrit[1].adjoint() + base_qutrit[1]* base_qutrit[0].adjoint()),
    3: (I*sqrt(p01))*density(base_qutrit[2])
}

B_13_3 = {
    1: sqrt(1 - p02)*eye(3),
    2: sqrt(p02)*(base_qutrit[0]* base_qutrit[2].adjoint() + base_qutrit[2]* base_qutrit[0].adjoint()),
    3: (I*sqrt(p02))*density(base_qutrit[1])
}

B_23_3 = {
    1: sqrt(1 - p12)*eye(3),
    2: sqrt(p12)*(base_qutrit[1]* base_qutrit[2].adjoint() + base_qutrit[2]* base_qutrit[1].adjoint()),
    3: (I*sqrt(p12))*density(base_qutrit[0])
}

Bs = [
    Matrix([
        [sqrt(1 - p01 - p02), 0, 0],
        [0, sqrt(1 - p01 - p12), 0],
        [0, 0, sqrt(1 - p02 - p12)]
    ]),
    sqrt(p01)*base_su3[1],
    sqrt(p02)*base_su3[4],
    sqrt(p12)*base_su3[6]
]


B_23_12 = list(TensorProduct(i,j) for i, j in product(B_2.values(), B_12_3.values()))
B_23_13 = list(TensorProduct(eye(2),j) for j in B_13_3.values())
B_23_23 = list(TensorProduct(eye(2),j) for j in B_23_3.values())

B_33_12 = list(TensorProduct(i,j) for i, j in product(B_12_3.values(), B_12_3.values()))
B_33_13 = list(TensorProduct(i,j) for i, j in product(B_13_3.values(), B_13_3.values()))
B_33_23 = list(TensorProduct(i,j) for i, j in product(B_23_3.values(), B_23_3.values()))


def bit_flip_sub_channels(rho, make_real):
    x, _ = rho.shape

    if x == 6:
        BFs = [B_23_12, B_23_13, B_23_23]
    elif x == 9:
        BFs = [B_33_12, B_33_13, B_33_23]

    results = []
    for BF in BFs: 
        result = apply_from_kraus(BF, rho).subs([(a.conjugate(), a) for a in make_real])
        result = result.applyfunc(lambda x: x.factor())
        results.append(result)
    
    return results


def bit_flip(rho, make_real):
    x, _ = rho.shape

    if x == 6:
        BFs = [B_23_12, B_23_13, B_23_23]
    elif x == 9:
        BFs = [B_33_12, B_33_13, B_33_23]

    result = Matrix.zeros(x)
    for BF in BFs: 
        result += S(1)/3*apply_from_kraus(BF, rho).subs([(a.conjugate(), a) for a in make_real]).applyfunc(lambda x: x.factor())
    
    return result

from ..correlations import negativity

def get_bit_flip_neg(rho, make_real):
    x, _ = rho.shape

    # Flips Individuales
    if x == 6:
        neg_01 = negativity(apply_from_kraus((TensorProduct(i,j) for i, j in product(B_2.values(), B_12_3.values())), rho))
        neg_02 = negativity(apply_from_kraus((TensorProduct(eye(2), j) for j in B_13_3.values()), rho))
        neg_03 = negativity(apply_from_kraus((TensorProduct(eye(2), j) for j in B_23_3.values()), rho))
    elif x == 9:
        neg_01 = negativity(apply_from_kraus((TensorProduct(i,j) for i, j in product(B_12_3.values(), B_12_3.values())), rho))
        neg_02 = negativity(apply_from_kraus((TensorProduct(i,j) for i, j in product(B_13_3.values(), B_13_3.values())), rho))
        neg_03 = negativity(apply_from_kraus((TensorProduct(i,j) for i, j in product(B_23_3.values(), B_23_3.values())), rho))
        
    # Flip General
    general_bit_flip_three = list(i/3 for i in B_23_3.values()) + list(i/3 for i in B_13_3.values()) + list(i/3 for i in B_12_3.values())
    if x == 6:
        neg_gen = negativity(apply_from_kraus((TensorProduct(i,j) for i, j in product(B_2.values(), general_bit_flip_three)), rho)).subs([(a.conjugate(), a) for a in make_real])
    elif x== 9:
        neg_gen = negativity(apply_from_kraus((TensorProduct(i,j) for i, j in product(general_bit_flip_three, general_bit_flip_three)), rho)).subs([(a.conjugate(), a) for a in make_real])
        
    return neg_01, neg_02, neg_03, neg_gen
        