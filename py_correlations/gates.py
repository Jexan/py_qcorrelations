from sympy import sqrt, Matrix
from . import gen_basis

__base_4__ = gen_basis(4)


H = 1/sqrt(2)*Matrix([[1, 1], [1, -1]])
CNOT = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


bell = [
    1/sqrt(2) * (__base_4__[0] + __base_4__[3]),
    1/sqrt(2) * (__base_4__[0] - __base_4__[3]),
    1/sqrt(2) * (__base_4__[1] + __base_4__[2]),
    1/sqrt(2) * (__base_4__[1] - __base_4__[2]),
]