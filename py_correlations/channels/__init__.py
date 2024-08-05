from sympy import Matrix

apply_from_kraus = lambda ks, ps: sum((i*ps*i.adjoint() for i in ks), Matrix.zeros(ps.shape[0]))

closure_condition = lambda ks: sum((i.adjoint()*i for i in ks), Matrix.zeros(ks[0].shape[0])) == eye(ks[0].shape[0])