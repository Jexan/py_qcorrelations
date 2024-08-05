from sympy import Matrix, eye, I

s1 = Matrix([
    [0, 1],
    [1, 0]
])

s2 = Matrix([
    [0, -I],
    [I, 0]
])

s3 = Matrix([
    [1, 0],
    [0, -1]
])

base_su2 = (eye(2), s1, s2, s3)