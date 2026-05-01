import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_


class Deter(nn.Module):
    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, act="SiLU"):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act()
        )
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}", BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action):
        """Deterministic state transition (block-GRU style)."""
        # (B, S, K), (B, D), (B, A)
        B = action.shape[0]

        # Flatten stochastic state and normalize action magnitude.
        # (B, S*K)
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        # (B, U)
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)

        # Concatenate projected inputs and broadcast over blocks.
        # (B, 3*U)
        x = torch.cat([x0, x1, x2], -1)
        # (B, G, 3*U)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)

        # Combine per-block deterministic state with per-block inputs.
        # (B, G, D/G + 3*U) -> (B, D + 3*U*G)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))

        # (B, D)
        x = self._dyn_hid(x)
        # (B, 3*D)
        x = self._dyn_gru(x)

        # Split GRU-style gates block-wise.
        # (B, G, 3*D/G)
        gates = torch.chunk(self.flat2group(x), 3, dim=-1)

        # (B, D)
        reset, cand, update = (self.group2flat(x) for x in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        # (B, D)
        return update * cand + (1 - update) * deter


class S4Backbone(nn.Module):
    """Diagonal continuous-time SSM-inspired transition."""

    def __init__(self, deter, stoch, act_dim, hidden, act="SiLU"):
        super().__init__()
        act_cls = getattr(torch.nn, act)
        self._inp = nn.Sequential(
            nn.Linear(stoch + act_dim, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32),
            act_cls(),
            nn.Linear(hidden, deter, bias=True),
        )
        self._log_dt = nn.Parameter(torch.zeros(deter))
        self._a_log = nn.Parameter(torch.zeros(deter))
        self._b = nn.Parameter(torch.ones(deter))
        self._c = nn.Parameter(torch.ones(deter))
        self._skip = nn.Parameter(torch.zeros(deter))
        self._norm = nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32)

    def forward(self, stoch, deter, action):
        bsz = action.shape[0]
        dtype = deter.dtype
        stoch = stoch.reshape(bsz, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        u = self._inp(torch.cat([stoch, action], dim=-1))
        dt = torch.nn.functional.softplus(self._log_dt).unsqueeze(0) + 1e-4
        a = -torch.nn.functional.softplus(self._a_log).unsqueeze(0)
        exp_a = torch.exp(a * dt)
        # Keep state update dtype aligned with recurrent state under AMP/compile.
        u = u.to(dtype)
        exp_a = exp_a.to(dtype)
        b = self._b.unsqueeze(0).to(dtype)
        c = self._c.unsqueeze(0).to(dtype)
        skip = self._skip.unsqueeze(0).to(dtype)
        state = exp_a * deter + (1.0 - exp_a) * (b * u)
        out = c * state + skip * deter
        return self._norm(out)


class S5Backbone(nn.Module):
    """S5-like transition with diagonal SSM and low-rank state mixing."""

    def __init__(self, deter, stoch, act_dim, hidden, act="SiLU", rank=16):
        super().__init__()
        act_cls = getattr(torch.nn, act)
        self._inp = nn.Sequential(
            nn.Linear(stoch + act_dim, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32),
            act_cls(),
            nn.Linear(hidden, deter, bias=True),
        )
        self._u = nn.Linear(deter, rank, bias=False)
        self._v = nn.Linear(rank, deter, bias=False)
        self._a_log = nn.Parameter(torch.zeros(deter))
        self._dt = nn.Parameter(torch.zeros(deter))
        self._gate = nn.Linear(2 * deter, deter, bias=True)
        self._cand = nn.Linear(2 * deter, deter, bias=True)
        self._norm = nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32)

    def forward(self, stoch, deter, action):
        bsz = action.shape[0]
        dtype = deter.dtype
        stoch = stoch.reshape(bsz, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        inp = self._inp(torch.cat([stoch, action], dim=-1))
        mixed = deter + self._v(self._u(deter))
        dt = torch.nn.functional.softplus(self._dt).unsqueeze(0) + 1e-4
        a = -torch.nn.functional.softplus(self._a_log).unsqueeze(0)
        # Keep state update and gate input dtype aligned under AMP/compile.
        inp = inp.to(dtype)
        mixed = mixed.to(dtype)
        dt = dt.to(dtype)
        a = a.to(dtype)
        ssm = torch.exp(a * dt) * mixed + dt * inp
        joint = torch.cat([ssm, inp], dim=-1)
        joint = joint.to(dtype)
        gate = torch.sigmoid(self._gate(joint))
        cand = torch.tanh(self._cand(joint))
        out = gate * cand + (1.0 - gate) * deter
        return self._norm(out)


class Mamba2Backbone(nn.Module):
    """Selective state-space style transition with input-dependent dynamics."""

    def __init__(self, deter, stoch, act_dim, hidden, act="SiLU"):
        super().__init__()
        act_cls = getattr(torch.nn, act)
        self._token = nn.Sequential(
            nn.Linear(stoch + act_dim, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32),
            act_cls(),
            nn.Linear(hidden, deter, bias=True),
        )
        self._a_base = nn.Parameter(torch.zeros(deter))
        self._dt_proj = nn.Linear(deter, deter, bias=True)
        self._b_proj = nn.Linear(deter, deter, bias=True)
        self._c_proj = nn.Linear(deter, deter, bias=True)
        self._z_proj = nn.Linear(deter, deter, bias=True)
        self._skip = nn.Parameter(torch.zeros(deter))
        self._norm = nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32)

    def forward(self, stoch, deter, action):
        bsz = action.shape[0]
        dtype = deter.dtype
        stoch = stoch.reshape(bsz, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        x = self._token(torch.cat([stoch, action], dim=-1))
        dt = torch.nn.functional.softplus(self._dt_proj(x)) + 1e-4
        a = -torch.nn.functional.softplus(self._a_base).unsqueeze(0)
        b = torch.tanh(self._b_proj(x))
        c = self._c_proj(x)
        z = torch.sigmoid(self._z_proj(x))
        # Keep selective state-space update dtype aligned under AMP/compile.
        x = x.to(dtype)
        dt = dt.to(dtype)
        a = a.to(dtype)
        b = b.to(dtype)
        c = c.to(dtype)
        z = z.to(dtype)
        skip = self._skip.unsqueeze(0).to(dtype)
        state = torch.exp(a * dt) * deter + dt * b * x
        y = c * state + skip * x
        out = z * y + (1.0 - z) * deter
        return self._norm(out)


class RelPosSelfAttention(nn.Module):
    def __init__(self, dim, nhead=8, max_rel_dist=256):
        super().__init__()
        assert dim % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = int(nhead)
        self.head_dim = dim // self.nhead
        self.scale = self.head_dim**-0.5
        self.max_rel_dist = int(max_rel_dist)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.rel_bias = nn.Embedding(2 * self.max_rel_dist + 1, self.nhead)

    def _rel_pos_bias(self, q_len, k_len, device):
        q_pos = torch.arange(q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)
        rel = (q_pos - k_pos).clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        bias = self.rel_bias(rel)  # (Q, K, H)
        return bias.permute(2, 0, 1)  # (H, Q, K)

    def forward(self, x):
        # x: (B, L, D)
        bsz, seqlen, dim = x.shape
        q = self.q_proj(x).reshape(bsz, seqlen, self.nhead, self.head_dim).transpose(1, 2)  # (B, H, L, Hd)
        k = self.k_proj(x).reshape(bsz, seqlen, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(bsz, seqlen, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        scores = scores + self._rel_pos_bias(seqlen, seqlen, x.device).unsqueeze(0)
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)  # (B, H, L, Hd)
        out = out.transpose(1, 2).reshape(bsz, seqlen, dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, ff_dim, nhead=8, max_rel_dist=256):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)
        self.attn = RelPosSelfAttention(dim, nhead=nhead, max_rel_dist=max_rel_dist)
        self.norm2 = nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim, bias=True),
            nn.GELU(),
            nn.Linear(ff_dim, dim, bias=True),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerBackbone(nn.Module):
    """Transformer-XL style transition with relative position and memory cache."""

    def __init__(self, deter, stoch, act_dim, hidden, act="SiLU", layers=2, mem_len=16, max_rel_dist=256):
        super().__init__()
        act_cls = getattr(torch.nn, act)
        self._mem_len = int(mem_len)
        self._inp = nn.Sequential(
            nn.Linear(stoch + act_dim, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32),
            act_cls(),
            nn.Linear(hidden, deter, bias=True),
        )
        self._layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=deter,
                    ff_dim=max(4 * deter, 512),
                    nhead=8,
                    max_rel_dist=max_rel_dist,
                )
                for _ in range(int(layers))
            ]
        )
        self._norm = nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32)

    def forward(self, stoch, deter, action, memory=None, reset=None):
        bsz = action.shape[0]
        stoch = stoch.reshape(bsz, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        token_inp = self._inp(torch.cat([stoch, action], dim=-1))
        # Query with current token and optionally attend to cached memory.
        cur = token_inp.unsqueeze(1)  # (B, 1, D)
        if memory is None:
            seq = cur
        else:
            if reset is not None:
                memory = torch.where(rpad(reset, memory.dim() - int(reset.dim())), torch.zeros_like(memory), memory)
            seq = torch.cat([memory, cur], dim=1)
        for layer in self._layers:
            seq = layer(seq)
        new_deter = self._norm(seq[:, -1] + deter)
        new_mem = seq[:, -self._mem_len :].detach()
        return new_deter, new_mem


class RSSM(nn.Module):
    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        self._backbone = str(getattr(config, "backbone", "gru")).lower()
        self._txl_mem_len = int(getattr(config, "txl_mem_len", 16))
        self._txl_memory = None
        if self._backbone == "gru":
            self._deter_net = Deter(
                self._deter,
                self.flat_stoch,
                act_dim,
                self._hidden,
                blocks=self._blocks,
                dynlayers=self._dyn_layers,
                act=config.act,
            )
        elif self._backbone == "s4":
            self._deter_net = S4Backbone(self._deter, self.flat_stoch, act_dim, self._hidden, act=config.act)
        elif self._backbone == "s5":
            self._deter_net = S5Backbone(self._deter, self.flat_stoch, act_dim, self._hidden, act=config.act)
        elif self._backbone == "mamba2":
            self._deter_net = Mamba2Backbone(self._deter, self.flat_stoch, act_dim, self._hidden, act=config.act)
        elif self._backbone in ("transformer", "transformer_cpc"):
            self._deter_net = TransformerBackbone(
                self._deter,
                self.flat_stoch,
                act_dim,
                self._hidden,
                act=config.act,
                layers=max(self._dyn_layers, 2),
                mem_len=self._txl_mem_len,
            )
        else:
            raise NotImplementedError(f"Unknown rssm backbone: {self._backbone}")

        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )

        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )
        self.apply(weight_init_)

    def clear_cache(self):
        """Clear transformer segment cache (no-op for non-transformer backbones)."""
        self._txl_memory = None

    def initial(self, batch_size):
        """Return an initial latent state."""
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations."""
        # (B, T, E), (B, T, A), ((B, S, K), (B, D)) (B, T)
        L = action.shape[1]
        stoch, deter = initial
        self.clear_cache()
        stochs, deters, logits = [], [], []
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        return stochs, deters, logits

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        """Single posterior step."""
        # (B, S, K), (B, D), (B, A), (B, E), (B,)
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )

        # Deterministic transition then posterior logits conditioned on embed.
        # (B, D)
        if self._backbone in ("transformer", "transformer_cpc"):
            deter, self._txl_memory = self._deter_net(
                stoch,
                deter,
                prev_action,
                memory=self._txl_memory,
                reset=reset,
            )
        else:
            deter = self._deter_net(stoch, deter, prev_action)
        # (B, D + E)
        x = torch.cat([deter, embed], dim=-1)
        # (B, S, K)
        logit = self._obs_net(x)

        # Sample discrete stochastic state via straight-through Gumbel-Softmax.
        # (B, S, K)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action):
        """Single prior step (no observation)."""

        # (B, D)
        if self._backbone in ("transformer", "transformer_cpc"):
            deter, self._txl_memory = self._deter_net(stoch, deter, prev_action, memory=self._txl_memory, reset=None)
        else:
            deter = self._deter_net(stoch, deter, prev_action)
        # (B, S, K)
        stoch, _ = self.prior(deter)
        return stoch, deter

    def prior(self, deter):
        """Compute prior distribution parameters and sample stoch."""

        # (B, S, K)
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions):
        """Roll out prior dynamics given a sequence of actions."""
        # (B, S, K), (B, D), (B, T, A)
        L = actions.shape[1]
        self.clear_cache()
        stochs, deters = [], []
        for i in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        # (B, T, S, K), (B, T, D)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        return stochs, deters

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        # (B, S, K), (B, D)
        # (B, S*K)
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        # (B, S*K + D)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        # Clipped gradients are not backpropagated using torch.clip.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss
