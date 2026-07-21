import numpy as np

def alpha_star_exact(n, eps):
    # n*alpha^2 - (2n-1)*alpha + (1/eps - 1) = 0
    a = n
    b = -(2*n-1)
    cc = (1/eps - 1)
    disc = b**2 - 4*a*cc
    root = (-b - np.sqrt(disc)) / (2*a)
    return root

def alpha_star_leading(n, eps):
    return (1-eps)/(2*n*eps)

print(f"{'n':>8} {'eps':>6} {'alpha*_exact':>14} {'alpha*_leading':>14} {'reldiff':>10}")
for n in [1e3, 1e4, 1e5]:
    for eps in [0.01, 0.05, 0.10]:
        ae = alpha_star_exact(n, eps)
        al = alpha_star_leading(n, eps)
        print(f"{n:8.0e} {eps:6.2f} {ae:14.6e} {al:14.6e} {abs(ae-al)/ae:10.4f}")
