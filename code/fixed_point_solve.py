import sympy as sp

alpha, n = sp.symbols('alpha n', positive=True)
k, m = sp.symbols('k m')  # Sigma_inf = k*Sigma, V_inf = m*Sigma

c = (n-1)/n

# eq1: k = alpha*c + (1-alpha)*c*k + alpha*(1-alpha)*m
eq1 = sp.Eq(k, alpha*c + (1-alpha)*c*k + alpha*(1-alpha)*m)
# eq2: m*alpha*(2-alpha) = alpha/n + (1-alpha)/n*k
eq2 = sp.Eq(m*alpha*(2-alpha), alpha/n + (1-alpha)/n*k)

sol = sp.solve([eq1, eq2], [k, m], dict=True)
print("Exact solution:")
for s in sol:
    ksol = sp.simplify(s[k])
    msol = sp.simplify(s[m])
    print("k (Sigma_inf/Sigma) =", ksol)
    print("m (Var[mu_inf]/Sigma) =", msol)

k_exact = sol[0][k]
m_exact = sol[0][m]

# Leading order in 1/n (large n expansion), alpha fixed in (0,1]
k_series = sp.series(k_exact, n, sp.oo, 3).removeO()
m_series = sp.series(m_exact, n, sp.oo, 3).removeO()
print("\nLarge-n series:")
print("k ~", sp.simplify(k_series))
print("m ~", sp.simplify(m_series))

# check alpha=0 limits (should recover pure decay / martingale limit)
print("\nAt alpha->0 limits (careful, m,k solved assuming alpha>0 fixed point exists):")
print("k(alpha->0):", sp.limit(k_exact, alpha, 0))
print("m(alpha->0):", sp.limit(m_exact, alpha, 0))

# simplify k ignoring the feedback (drop alpha(1-alpha)*m term) -- leading order approx
k_naive = sp.symbols('k_naive')
eq1_naive = sp.Eq(k_naive, alpha*c + (1-alpha)*c*k_naive)
k_naive_sol = sp.solve(eq1_naive, k_naive)[0]
k_naive_sol = sp.simplify(k_naive_sol)
print("\nNaive (decoupled) k, ignoring V feedback:")
print("k_naive =", k_naive_sol)
print("k_naive large-n:", sp.simplify(sp.series(k_naive_sol, n, sp.oo, 3).removeO()))

# compare k_exact vs k_naive numerically
import numpy as np
for nv in [200, 2000]:
    for av in [0.01,0.05,0.2,1.0]:
        kx = float(k_exact.subs({alpha:av, n:nv}))
        kn = float(k_naive_sol.subs({alpha:av, n:nv}))
        print(f"n={nv} alpha={av}: k_exact={kx:.6f} k_naive={kn:.6f} reldiff={(abs(kx-kn)/kx):.2e}")
