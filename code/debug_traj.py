import torch, numpy as np
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0); np.random.seed(0)

d=5
mu_true = torch.arange(1,d+1,dtype=torch.float64,device=device)
sigma2_true = torch.tensor([0.5,1.0,2.0,3.0,5.0],dtype=torch.float64,device=device)
Sigma_true = torch.diag(sigma2_true)
L_true = torch.linalg.cholesky(Sigma_true)

def theory_trajectory(alpha,n,sigma2,T):
    c=(n-1)/n
    Et=sigma2; vt=0.0
    Es=[Et]; vs=[vt]
    for _ in range(T):
        E_new = alpha*c*sigma2 + (1-alpha)*c*Et + alpha*(1-alpha)*vt
        v_new = (alpha/n)*sigma2 + ((1-alpha)/n)*Et + (1-alpha)**2*vt
        Et,vt=E_new,v_new
        Es.append(Et); vs.append(vt)
    return np.array(Es), np.array(vs)

def simulate(alpha,n,T,R,device):
    n_r=int(round(alpha*n)); n_s=n-n_r
    mu_t = mu_true.unsqueeze(0).repeat(R,1).clone()
    Sigma_t = Sigma_true.unsqueeze(0).repeat(R,1,1).clone()
    mu_hist=torch.empty((T+1,R,d),device=device)
    Sigma_hist_mean=torch.empty((T+1,d,d),device=device)
    mu_hist[0]=mu_t; Sigma_hist_mean[0]=Sigma_t.mean(0)
    for t in range(1,T+1):
        parts=[]
        if n_r>0:
            z=torch.randn(R,n_r,d,device=device)
            parts.append(mu_true.view(1,1,d)+torch.einsum('ij,rnj->rni',L_true,z))
        if n_s>0:
            L=torch.linalg.cholesky(Sigma_t)
            z=torch.randn(R,n_s,d,device=device)
            parts.append(mu_t.unsqueeze(1)+torch.einsum('rij,rnj->rni',L,z))
        X=torch.cat(parts,dim=1)
        mu_new=X.mean(1)
        Xc=X-mu_new.unsqueeze(1)
        Sigma_new=torch.einsum('rni,rnj->rij',Xc,Xc)/n
        mu_t,Sigma_t=mu_new,Sigma_new
        mu_hist[t]=mu_t; Sigma_hist_mean[t]=Sigma_t.mean(0)
    return Sigma_hist_mean.cpu().numpy(), mu_hist.cpu().numpy()

n=200; alpha=0.0; T=200; R=2000
Sigma_mean, mu_hist = simulate(alpha,n,T,R,device)
diag=np.arange(d)
sim_Sigma = Sigma_mean[:,diag,diag]
sim_varmu = mu_hist.var(axis=1,ddof=1)

for k in range(d):
    sigma2_k=sigma2_true[k].item()
    Es,vs = theory_trajectory(alpha,n,sigma2_k,T)
    print(f"--- coord {k}, sigma2={sigma2_k} ---")
    for tcp in [20,40,60,80,100,120,140,160,180,200]:
        th_S=Es[tcp]; sim_S=sim_Sigma[tcp,k]
        th_V=vs[tcp]; sim_V=sim_varmu[tcp,k]
        print(f"  t={tcp:4d}  Sigma: th={th_S:.6f} sim={sim_S:.6f} relerr={abs(sim_S-th_S)/th_S:.4f}   Vmu: th={th_V:.6f} sim={sim_V:.6f} relerr={abs(sim_V-th_V)/max(th_V,1e-12):.4f}")
