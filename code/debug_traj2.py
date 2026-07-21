import torch, numpy as np
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1); np.random.seed(1)

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

def simulate(alpha,n,T,R,device,checkpoints):
    n_r=int(round(alpha*n)); n_s=n-n_r
    mu_t = mu_true.unsqueeze(0).repeat(R,1).clone()
    Sigma_t = Sigma_true.unsqueeze(0).repeat(R,1,1).clone()
    out_S = {}
    out_V = {}
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
        if t in checkpoints:
            diag=torch.arange(d)
            Sdiag = Sigma_t[:,diag,diag].cpu().numpy()  # (R,d)
            out_S[t] = Sdiag
            out_V[t] = mu_t.cpu().numpy()  # (R,d), to compute var later
    return out_S, out_V

n=200; alpha=0.0; T=200
checkpoints=[100,200]
R=100000  # much larger replicate count for this targeted check
S_by_t, mu_by_t = simulate(alpha,n,T,R,device,checkpoints)

for tcp in checkpoints:
    Sdiag = S_by_t[tcp]  # (R,d)
    mus = mu_by_t[tcp]
    for k in range(d):
        sigma2_k = sigma2_true[k].item()
        Es,vs = theory_trajectory(alpha,n,sigma2_k,T)
        th_S = Es[tcp]
        samp = Sdiag[:,k]
        sim_mean = samp.mean()
        sim_std = samp.std(ddof=1)
        stderr = sim_std/np.sqrt(R)
        n_stderr = (sim_mean-th_S)/stderr
        th_V = vs[tcp]
        sim_V = mus[:,k].var(ddof=1)
        print(f"t={tcp} coord={k} sigma2={sigma2_k}: theory_S={th_S:.5f} sim_S={sim_mean:.5f} rel_err={abs(sim_mean-th_S)/th_S:.4f} stderr={stderr:.5f} (#stderr={n_stderr:.2f})  skew_check(sim_std/mean)={sim_std/sim_mean:.3f}   theory_V={th_V:.5f} sim_V={sim_V:.5f} relerrV={abs(sim_V-th_V)/max(th_V,1e-12):.4f}")
