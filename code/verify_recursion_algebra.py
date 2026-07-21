import sympy as sp

n_r, n_s, n, alpha, Sigma, ESprev, Vprev = sp.symbols('n_r n_s n alpha Sigma ESprev Vprev', positive=True)

# n*E[Sigma_t] = (n_r-1)*Sigma + (n_s-1)*E[Sigma_{t-1}] + (n_r*n_s/n)*(V_{t-1} + Sigma/n_r + E[Sigma_{t-1}]/n_s)
lhs_n_ESigma_t = (n_r-1)*Sigma + (n_s-1)*ESprev + (n_r*n_s/n)*(Vprev + Sigma/n_r + ESprev/n_s)
lhs_n_ESigma_t = sp.expand(lhs_n_ESigma_t)
print("n*E[Sigma_t] (in terms of n_r,n_s,n) =")
print(lhs_n_ESigma_t)

# substitute n_r = alpha*n, n_s=(1-alpha)*n
sub = {n_r: alpha*n, n_s:(1-alpha)*n}
expr = lhs_n_ESigma_t.subs(sub)
expr = sp.simplify(expr)
print("\nAfter substituting n_r=alpha n, n_s=(1-alpha) n:")
print(sp.expand(expr))

# Divide by n to get E[Sigma_t]
ESigma_t = sp.simplify(expr / n)
ESigma_t = sp.expand(ESigma_t)
print("\nE[Sigma_t] =")
print(ESigma_t)

# compare to my claimed form: alpha*c*Sigma + (1-alpha)*c*ESprev + alpha*(1-alpha)*Vprev, c=(n-1)/n
c = (n-1)/n
claimed = alpha*c*Sigma + (1-alpha)*c*ESprev + alpha*(1-alpha)*Vprev
claimed = sp.expand(claimed)
print("\nClaimed form =")
print(claimed)

diff = sp.simplify(ESigma_t - claimed)
print("\nDifference (should be 0 if correct):", diff)
