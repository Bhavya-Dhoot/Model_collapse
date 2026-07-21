import sympy as sp

alpha, n, eps = sp.symbols('alpha n epsilon', positive=True)

k_exact = alpha*(alpha*n - 2*n + 1)/(alpha**2*n - 2*alpha*n + alpha - 1)
k_exact_factored = sp.factor(k_exact)
print("k_exact factored:", k_exact_factored)

# try the manual factoring form
k_manual = alpha*(n*(2-alpha)-1) / ((1-alpha) + n*alpha*(2-alpha))
print("check manual == exact:", sp.simplify(k_manual - k_exact)==0)

m_exact = -1/(alpha**2*n - 2*alpha*n + alpha - 1)
m_manual = 1/((1-alpha) + n*alpha*(2-alpha))
print("check m manual == exact:", sp.simplify(m_manual - m_exact)==0)

print()
print("FINAL CLOSED FORMS:")
print("Sigma_inf/Sigma = k =", sp.nsimplify(k_manual))
print("Var[mu_inf]/Sigma = m =", sp.nsimplify(m_manual))

# Now solve threshold: 1 - k = eps  =>  k = 1-eps
alpha_star_eq = sp.Eq(k_manual, 1-eps)
alpha_sols = sp.solve(alpha_star_eq, alpha)
print("\nExact alpha* solutions (root of quadratic):")
for a in alpha_sols:
    print(sp.simplify(a))

# Leading order approx: assume alpha small, n large, alpha*n = O(1) or so.
# k = alpha*(n(2-alpha)-1)/((1-alpha)+n*alpha*(2-alpha))
# For alpha small: numerator ~ alpha*(2n-1) ~ 2*alpha*n ; denominator ~ 1 + 2*alpha*n
# k ~ 2*alpha*n/(1+2*alpha*n)  => 1-k ~ 1/(1+2*alpha*n) = eps
# => 1+2*alpha*n = 1/eps => alpha* ~ (1/eps - 1)/(2n) = (1-eps)/(2*n*eps)
alpha_star_leading = (1-eps)/(2*n*eps)
print("\nLeading-order (small alpha, large n) alpha* ~ (1-eps)/(2*n*eps)")

# numeric comparison of leading order vs exact solve
import numpy as np
for nv in [1e3,1e4,1e5]:
    for ev in [0.01,0.05,0.10]:
        # exact: solve numerically
        f = sp.lambdify(alpha, k_manual.subs({n:nv}) - (1-ev))
        from scipy.optimize import brentq
        try:
            a_exact = brentq(f, 1e-12, 1-1e-9)
        except Exception as e:
            a_exact = None
        a_lead = (1-ev)/(2*nv*ev)
        print(f"n={nv:.0e} eps={ev}: alpha*_exact={a_exact:.6e}  alpha*_leading={a_lead:.6e}  reldiff={abs(a_exact-a_lead)/a_exact:.3e}")
