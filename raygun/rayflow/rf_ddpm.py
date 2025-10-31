import torch
import torch.nn as nn
from raygun.modelv2.model_utils import Block
from einops import rearrange
import torch.nn.functional as F

def subsampled(betas, N, gamma=3.0):
    alphas    = (1-betas).cumprod(dim=0)
    timesteps = powerlaw_subsample(len(alphas), N+1, gamma)
    ab        = alphas[timesteps]
    cbetat    = 1 - (ab[1:] / ab[:-1])
    return cbetat


def get_time_embedding(
    time_steps: torch.Tensor,
    t_emb_dim: int
) -> torch.Tensor: 
    assert t_emb_dim%2 == 0, "time embedding must be divisible by 2."
    
    factor = 2 * torch.arange(start = 0, 
                              end = t_emb_dim//2, 
                              dtype=torch.float32, 
                              device=time_steps.device
                             ) / (t_emb_dim)
    
    factor = 10000**factor

    t_emb  = time_steps[:,None] # B -> (B, 1) 
    t_emb  = t_emb/factor # (B, 1) -> (B, t_emb_dim//2)
    t_emb  = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1) # (B , t_emb_dim)
    
    return t_emb


class DiffusionProcess:
    def __init__(self, betas):
        # Precomputing beta, alpha, and alpha_bar for all t's.
        self.N               = len(betas)
        self.betas           = betas.clone()
        self.alpha_bars      = (1-betas).cumprod(dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-self.alpha_bars)
        
    def add_noise(self, original, noise, t):
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bars.to(original.device)[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.to(original.device)[t]
        
        # Broadcast to multiply with the original image.
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None]
        
        # Return
        return ((sqrt_alpha_bar_t * original) \
                           + \
               (sqrt_one_minus_alpha_bar_t * noise)).float()
    
    
    def sample_prev_timestep(self, xt, noise_pred, 
                             a_curr, a_next, eps=1e-3):
        device = xt.device
        a_curr = torch.clamp(a_curr, min=eps).to(device)
        a_next = torch.clamp(a_next, min=eps).to(device)
        x0pred = (xt - torch.sqrt(1-a_curr) * noise_pred) / torch.sqrt(a_curr)
        xtmin1 = (x0pred * torch.sqrt(a_next) + torch.sqrt(1-a_next)*noise_pred)
        return xtmin1

        
class FiLM(nn.Module):
    def __init__(self, 
                 fixed_dim=50, 
                 embed_dim=1280):
        super().__init__()
        self.to_scale_w = nn.Linear(embed_dim, embed_dim)
        self.to_shift_w = nn.Linear(embed_dim, embed_dim)
        self.to_scale_h = nn.Linear(embed_dim, fixed_dim)
        self.to_shift_h = nn.Linear(embed_dim, fixed_dim)

    def forward(self, x, cond):
        gamma_h         = self.to_scale_w(cond).unsqueeze(1)
        beta_h          = self.to_shift_w(cond).unsqueeze(1)
        x_h             = gamma_h * x + beta_h
        gamma_w         = rearrange(self.to_scale_h(cond), 
                                   "b w -> b w 1")
        beta_w          = rearrange(self.to_shift_h(cond), 
                                   "b w -> b w 1")
        return (1+gamma_w) * x_h + beta_w

    
class FlowBlock(nn.Module):
    def __init__(self, dim=1280,
                 fixed_dim=50,
                 convkernel=7,
                 nhead=20):
        super().__init__()
        self.block = Block(dim=dim, 
                          convkernel=convkernel,
                          attnheads=nhead)
        self.filmt = FiLM(fixed_dim, dim)
        self.films = FiLM(fixed_dim, dim)
        
    def forward(self, x_t, source_emb, 
                t_emb, sp_emb):
        y_t = self.block(x_t) + source_emb
        y_t = self.block(self.films(y_t, sp_emb)) \
              + source_emb
        return self.filmt(y_t, t_emb)
        

class RayFlowDenoiser(nn.Module):
    def __init__(self, 
                 betas, 
                 no_flowblock=5, 
                 embed_dim=1280, 
                 t_hidden_dim=640,
                 sp_emb_dim=640,
                 sp_st_dim=640,
                 sp_hidden_dim=640,
                 fixed_dim=50,
                 convkernel=7,
                 max_species=200,
                 nhead=20):
        super().__init__()
        assert embed_dim % 2 == 0, "The embedding dimensions should be even"
        self.diffproc  = DiffusionProcess(betas)
        self.T         = len(betas)
        
        self.time_proj = nn.Sequential(nn.Linear(embed_dim, 
                                                 t_hidden_dim), 
                                       nn.SiLU(),
                                       nn.Linear(t_hidden_dim, 
                                                 embed_dim))
        self.sp_proj  = nn.Sequential(nn.Linear(sp_st_dim, 
                                                sp_hidden_dim), 
                                      nn.SiLU(),
                                      nn.Linear(sp_hidden_dim, 
                                                embed_dim//2))
        self.emb_dim  = embed_dim
        self.sp_dim   = sp_st_dim
        self.fixed_dim= fixed_dim
        
        self.spembed  = nn.Sequential(nn.Embedding(max_species, 
                                                   sp_emb_dim),
                                      nn.SiLU(),
                                      nn.Linear(sp_emb_dim, sp_st_dim))
        self.sp_st_dim= sp_st_dim
        
        self.blocks   = nn.ModuleList()
        for i in range(no_flowblock):
            self.blocks.append(FlowBlock(dim=embed_dim, 
                                        fixed_dim=fixed_dim, 
                                        convkernel=convkernel, 
                                        nhead=nhead))
        return
    
    def compute_species_embed(self, sp):
        return self.spembed(sp)
    
    def forward(self, x_t,
                source_emb,
                source_sp, 
                target_sp, t):
        source_species = self.spembed(source_sp)
        target_species = self.spembed(target_sp)
        
        t_emb          = get_time_embedding(t, self.emb_dim) 
        t_emb          = self.time_proj(t_emb)
        
        s_sp_emb       = self.sp_proj(source_species)
        t_sp_emb       = self.sp_proj(target_species)
        
        sp_emb         = torch.concat([s_sp_emb, t_sp_emb],
                               dim=-1)
        y_t            = x_t
        for i, modx in enumerate(self.blocks):
            y_t        = modx(y_t, source_emb, 
                              t_emb, sp_emb)
        return y_t
    
    def compute_loss(self, 
                     start_sp_emb, 
                     start_sp_tok, 
                     end_sp_emb, 
                     end_sp_tok):
        
        bsize        = start_sp_tok.shape[0]
        t            = torch.randint(0, self.T, 
                                     [bsize,]).to(start_sp_emb.device)
        noise        = torch.randn_like(end_sp_emb)
        
        xt           = self.diffproc.add_noise(end_sp_emb, 
                                               noise, t)
        npred        = self.forward(xt, start_sp_emb, 
                                   start_sp_tok, 
                                   end_sp_tok, t)
        
        loss         = F.mse_loss(npred, noise)
        return loss
    
    
    @torch.no_grad()
    def sample(self, start_sp_emb, 
              start_sp_tok, 
              end_sp_emb, N=50,
               start=4):
        T            = self.T
        gamma        = 2
        timesteps    = powerlaw_subsample(T, N+start, gamma)[start:]
        alphabars    = self.diffproc.alpha_bars
        xt           = torch.randn_like(start_sp_emb)
        b, k, d      = start_sp_emb.shape
        dev          = start_sp_emb.device
        for i in reversed(range(N)):
            a_curr   = alphabars[timesteps[i]]
            a_next   = (alphabars[timesteps[i-1]] if i > 0 
                        else alphabars[0])
            ti       = torch.tensor([timesteps[i]
                                    ]).expand(b),to(dev)
            eps_pred = self.model(xt, start_sp_emb, 
                                  start_sp_tok, 
                                  end_sp_emb, ti)
            xt       = self.sample_prev_timestep(xt, eps_pred, 
                                                 a_curr, a_next)
        return xt
    
