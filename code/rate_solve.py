import sympy as sp

alpha, n = sp.symbols('alpha n', positive=True)
c = (n-1)/n

A = sp.Matrix([
    [(1-alpha)*c,        alpha*(1-alpha)],
    [(1-alpha)/n,        (1-alpha)**2   ]
])

eigs = A.eigenvals()
print("Eigenvalues of A (exact):")
for e,mult in eigs.items():
    print(sp.simplify(e), " mult=", mult)

# trace and det check
print("trace(A) =", sp.simplify(sp.trace(A)))
print("det(A) =", sp.simplify(A.det()))

# large n leading order of eigenvalues, alpha fixed
lam = list(eigs.keys())
for i,e in enumerate(lam):
    ser = sp.series(e, n, sp.oo, 2).removeO()
    print(f"lambda_{i} large-n leading order:", sp.simplify(ser))

# also check spectral radius numerically for a grid to see which eigenvalue dominates
import numpy as np
for nv in [200,2000]:
    for av in [0.01,0.05,0.2,1.0]:
        Anum = np.array(A.subs({alpha:av,n:nv})).astype(float)
        w = np.linalg.eigvals(Anum)
        print(f"n={nv} alpha={av}: eigs={w}, spec_radius={max(abs(w)):.6f}, (1-alpha)*c={(1-av)*(nv-1)/nv:.6f}")
