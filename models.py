import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module): 
    def __init__(self, embedding_size=1024, in_channels=3, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(in_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        if embedding_size == 1024:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(1024, embedding_size)

    def forward(self, observation):
        """
        observation: [B, C, H, W] or [B, T, C, H, W]
        """
        if observation.dim() == 5:
            B, T, C, H, W = observation.shape
            x = observation.reshape(B * T, C, H, W)
            hidden = self.act_fn(self.conv1(x))
            hidden = self.act_fn(self.conv2(hidden))
            hidden = self.act_fn(self.conv3(hidden))
            hidden = self.act_fn(self.conv4(hidden))
            hidden = hidden.view(B * T, -1)
            hidden = self.fc(hidden)
            return hidden.view(B, T, -1)
        else:
            hidden = self.act_fn(self.conv1(observation))
            hidden = self.act_fn(self.conv2(hidden))
            hidden = self.act_fn(self.conv3(hidden))
            hidden = self.act_fn(self.conv4(hidden))
            hidden = hidden.view(observation.size(0), -1)
            hidden = self.fc(hidden)
            return hidden


class ConvDecoder(nn.Module):
    """
    PlaNet Visual Decoder: reconstructs 64x64 images from state+latent.
    """
    def __init__(self, state_size, latent_size, embedding_size=1024, out_channels=3, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.out_channels = out_channels
        self.fc1 = nn.Linear(latent_size + state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, out_channels, 6, stride=2)

    def forward(self, state, latent=None):
        """
        state: [B, state_size] or [B, T, state_size]  (if latent is None, state is full z = [h, s])
        latent: [B, latent_size] or [B, T, latent_size] (optional)
        """
        if latent is not None:
            z = torch.cat([state, latent], dim=-1)
        else:
            z = state
        
        if z.dim() == 3:
            B, T, D = z.shape
            z_flat = z.reshape(B * T, D)
        else:
            B = z.size(0)
            T = None
            z_flat = z
            
        hidden = self.fc1(z_flat)
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        
        if T is not None:
            return observation.view(B, T, self.out_channels, observation.size(-2), observation.size(-1))
        return observation


class RSSM(nn.Module):
    """
    Recurrent State Space Model - PlaNet implementation.
    
    Key components:
    - Deterministic state update: h_t = GRU(act_fn(linear([s_{t-1}, a_{t-1}])), h_{t-1})
    - Prior: p(s_t | h_t) = N(mean, std)
    - Posterior: q(s_t | h_t, e_t) = N(mean, std)
    """
    def __init__(self, stoch_dim=30, deter_dim=200, act_dim=1, embed_dim=1024, hidden_dim=200, activation_function='relu'):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.act_dim = act_dim
        self.act_fn = getattr(F, activation_function)
        
        # GRU for deterministic state (takes transformed input, hidden state)
        self.grucell = nn.GRUCell(deter_dim, deter_dim)
        
        # Linear layer to transform [s, a] before GRU (key difference from incorrect impl)
        self.lat_act_layer = nn.Linear(stoch_dim + act_dim, deter_dim)
        
        # Prior network: h -> (mean, std) - single hidden layer
        self.fc_prior_1 = nn.Linear(deter_dim, hidden_dim)
        self.fc_prior_m = nn.Linear(hidden_dim, stoch_dim)
        self.fc_prior_s = nn.Linear(hidden_dim, stoch_dim)
        
        # Posterior network: [h, e] -> (mean, std) - single hidden layer  
        self.fc_posterior_1 = nn.Linear(deter_dim + embed_dim, hidden_dim)
        self.fc_posterior_m = nn.Linear(hidden_dim, stoch_dim)
        self.fc_posterior_s = nn.Linear(hidden_dim, stoch_dim)

    def init_state(self, embed=None, batch_size=None, device=None):
        """
        Initialize h and s to zeros.
        """
        if embed is not None:
            batch_size = embed.size(0)
            device = embed.device
        h0 = torch.zeros(batch_size, self.deter_dim, device=device)
        s0 = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h0, s0

    def deterministic_state_fwd(self, h_t, s_t, a_t):
        """
        Deterministic state transition.
        h_{t+1} = GRU(act_fn(linear([s_t, a_t])), h_t)
        """
        x = torch.cat([s_t, a_t], dim=-1)
        x = self.act_fn(self.lat_act_layer(x))
        return self.grucell(x, h_t)

    def state_prior(self, h_t, sample=False):
        """
        Prior distribution p(s_t | h_t).
        Returns (mean, std) or sample if sample=True.
        """
        z = self.act_fn(self.fc_prior_1(h_t))
        m = self.fc_prior_m(z)
        s = F.softplus(self.fc_prior_s(z)) + 0.1  # min_std = 0.1 as in reference
        if sample:
            return m + torch.randn_like(m) * s
        return m, s

    def state_posterior(self, h_t, e_t, sample=False):
        """
        Posterior distribution q(s_t | h_t, e_t).
        Returns (mean, std) or sample if sample=True.
        """
        z = torch.cat([h_t, e_t], dim=-1)
        z = self.act_fn(self.fc_posterior_1(z))
        m = self.fc_posterior_m(z)
        s = F.softplus(self.fc_posterior_s(z)) + 0.1  # min_std = 0.1 as in reference
        if sample:
            return m + torch.randn_like(m) * s
        return m, s

    def get_init_state(self, enc, h_t=None, s_t=None, a_t=None, mean=False):
        """
        Returns the initial posterior given the observation.
        Matches reference implementation signature.
        """
        N, dev = enc.size(0), enc.device
        h_t = torch.zeros(N, self.deter_dim).to(dev) if h_t is None else h_t
        s_t = torch.zeros(N, self.stoch_dim).to(dev) if s_t is None else s_t
        a_t = torch.zeros(N, self.act_dim).to(dev) if a_t is None else a_t
        h_tp1 = self.deterministic_state_fwd(h_t, s_t, a_t)
        if mean:
            s_tp1 = self.state_posterior(h_tp1, enc)
        else:
            s_tp1, _ = self.state_posterior(h_tp1, enc, sample=True)
        return h_tp1, s_tp1

    def observe(self, embeds, actions):
        """
        Process a sequence of embeddings and actions.
        embeds: [B, T, E]
        actions: [B, T, A]
        Returns dict with states and distribution parameters.
        """
        B, T, E = embeds.shape
        device = embeds.device
        
        # Initialize from zeros (reference uses e_t[0] for first posterior)
        h_t, s_t = self.init_state(batch_size=B, device=device)
        a_prev = torch.zeros(B, self.act_dim, device=device)
        
        states = []
        priors = []
        posteriors = []
        posterior_samples = []
        
        for t in range(T):
            # Deterministic state update
            h_t = self.deterministic_state_fwd(h_t, s_t, a_prev)
            states.append(h_t)
            
            # Prior and posterior
            priors.append(self.state_prior(h_t))
            posteriors.append(self.state_posterior(h_t, embeds[:, t]))
            
            # Sample from posterior
            post_mean, post_std = posteriors[-1]
            s_t = post_mean + torch.randn_like(post_std) * post_std
            posterior_samples.append(s_t)
            
            # Update action for next step
            a_prev = actions[:, t]
        
        # Stack results
        h_seq = torch.stack(states, dim=1)  # [B, T, deter_dim]
        s_seq = torch.stack(posterior_samples, dim=1)  # [B, T, stoch_dim]
        
        prior_means = torch.stack([p[0] for p in priors], dim=1)
        prior_stds = torch.stack([p[1] for p in priors], dim=1)
        post_means = torch.stack([p[0] for p in posteriors], dim=1)
        post_stds = torch.stack([p[1] for p in posteriors], dim=1)
        
        return {
            "h": h_seq,
            "s": s_seq,
            "prior_mean": prior_means,
            "prior_std": prior_stds,
            "post_mean": post_means,
            "post_std": post_stds,
        }

    def observe_step(self, embed, action, h, s, sample=True):
        """
        Single-step filter update.
        """
        h_next = self.deterministic_state_fwd(h, s, action)
        p_mean, p_std = self.state_prior(h_next)
        q_mean, q_std = self.state_posterior(h_next, embed)
        if sample:
            s_next = q_mean + torch.randn_like(q_std) * q_std
        else:
            s_next = q_mean
        return h_next, s_next, (p_mean, p_std), (q_mean, q_std)

    @torch.no_grad()
    def imagine_step(self, h, s, a):
        """
        One prior step for planning/imagination.
        """
        h_next = self.deterministic_state_fwd(h, s, a)
        s_next = self.state_prior(h_next, sample=True)
        return h_next, s_next


class RewardModel(nn.Module):
    def __init__(self, state_size=200, latent_size=30, hidden_dim=200, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc_reward_1 = nn.Linear(state_size + latent_size, hidden_dim)
        self.fc_reward_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_reward_3 = nn.Linear(hidden_dim, 1)

    def forward(self, h, s=None):
        """
        h: deterministic state [B, state_size] or full z=[h,s] if s is None
        s: stochastic state [B, latent_size] (optional)
        """
        if s is not None:
            x = torch.cat([h, s], dim=-1)
        else:
            x = h
        r = self.act_fn(self.fc_reward_1(x))
        r = self.act_fn(self.fc_reward_2(r))
        return self.fc_reward_3(r).squeeze(-1)

