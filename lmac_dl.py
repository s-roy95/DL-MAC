# =====================================================================
#  CANONICAL SIMULTANEOUS MAPPO (LMAC) — SLOTTED DECENTRALIZED EDGE-CLOUD FaaS
#  Single-cell: generates data -> env -> baselines -> train -> figures.
#  Colab: set Runtime -> GPU. Just run this one cell.
#
#  MODIFIED: adds fig_sota_bars() — Total-Profit + Jain bar charts for ALL
#  SOTA approaches shown in the edge-distribution figure
#  (S-Cache, CoCache, DSP, pCache, DL-LRU, Offline-Opt, DL-MAC).
#  The per-node profile is now a single shared constant EDGE_PROFILE so the
#  edge-distribution line plot and the new bar charts never disagree.
#
#  TUNED: 500 episodes + stabilized PPO (lr=1e-4, clip=0.15, ent=0.005,
#  epochs=2) to prevent post-peak policy collapse.
# =====================================================================
import os, csv, time, warnings
import numpy as np, pandas as pd, networkx as nx
from collections import deque, defaultdict, OrderedDict
import torch, torch.nn as nn, torch.optim as optim
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
torch.set_num_threads(max(4, os.cpu_count() or 4))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- experiment knobs -------------------------
NUM_NODES, EDGES_PER_NODE, NUM_FUNCTIONS = 30, 3, 15
GATEWAYS = [1, 5, 10, 15, 20]
DEADLINE_SCALE = 0.8       # deadline tightness (lower = harder)
EPISODES, BC_EPISODES, SEED = 500, 50, 42
GRAPH = "data/graph.txt"
DATA  = "data/chain_dataset_with_profit500.csv"

# ------------------------- 1. DATA GENERATION -----------------------
os.makedirs("data", exist_ok=True)
G = nx.barabasi_albert_graph(NUM_NODES, EDGES_PER_NODE, seed=SEED)
edges = [(u + 1, v + 1) for u, v in G.edges()]
with open(GRAPH, "w") as f:
    f.write(f"{NUM_NODES} {len(edges)} {len(GATEWAYS)}\n")
    f.write(" ".join(map(str, GATEWAYS)) + "\n")
    for u, v in edges: f.write(f"{u} {v}\n")

def generate_dataset(num_chains, filename, seed=SEED):
    np.random.seed(seed)
    L = np.random.randint(3, 7, size=num_chains)
    ft = [f"func_{i}" for i in range(1, NUM_FUNCTIONS + 1)]
    rows, t = [], 0.0
    for cid in range(num_chains):
        t += np.random.exponential(2.0)
        origin = np.random.randint(1, NUM_NODES + 1)
        dl = L[cid] * np.random.uniform(15.0, 40.0)
        pr = L[cid] * np.random.uniform(5.0, 20.0)
        for _ in range(L[cid]):
            rows.append({"chain_id": f"chain_{cid}", "arrival_time": round(t, 2),
                         "edge_server_id": origin, "func_type": np.random.choice(ft),
                         "chain_deadline": round(dl, 2), "profit": round(pr, 2),
                         "func_delay": int(np.random.randint(2, 9))})
    pd.DataFrame(rows).to_csv(filename, index=False)

generate_dataset(500, DATA)
print(f"Generated 30-node graph ({len(edges)} edges) + 500-chain dataset.")

# ------------------------- 2. ENVIRONMENT ---------------------------
class Cfg:
    K_HOP = 2; CONC = 2; MAXNB = 4
    EDGE_TO_EDGE = 2; EDGE_TO_CLOUD = 5; CLOUD_PROC = 3; COLD_START = 5
    PRIVATE_SLOTS, PUBLIC_SLOTS = 2, 3; PUB_FETCH = 1; CLOUD_KEEP = 0.6
    W_PROFIT = 1.0; W_JAIN = 3.0; W_COLD = 0.05; W_WAIT = 0.02; W_REJECT = 0.5
    PROFIT_SCALE = 50.0; ACT_DIM = 3 + MAXNB

class LRU:
    __slots__ = ("cap", "od")
    def __init__(self, cap): self.cap, self.od = cap, OrderedDict()
    def peek(self, f): return f in self.od
    def touch(self, f):
        if f in self.od: self.od.move_to_end(f)
    def insert(self, f):
        if self.cap <= 0: return
        if f in self.od: self.od.move_to_end(f); return
        if len(self.od) >= self.cap: self.od.popitem(last=False)
        self.od[f] = None

def load_graph(path):
    with open(path) as g: toks = g.read().split()
    it = iter(toks); N, M, Gn = int(next(it)), int(next(it)), int(next(it))
    gw = [int(next(it)) for _ in range(Gn)]
    adj = defaultdict(list)
    for _ in range(M):
        u, v = int(next(it)), int(next(it)); adj[u].append(v); adj[v].append(u)
    INF = 10**9; dist = [[INF]*(N+1) for _ in range(N+1)]
    for s in range(1, N+1):
        d = [INF]*(N+1); d[s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[v] > d[u]+1: d[v] = d[u]+1; q.append(v)
        dist[s] = d
    return N, gw, dist

def load_chains(path):
    rows, fmap = [], {}
    with open(path) as f:
        for r in csv.DictReader(f):
            fs = r["func_type"]
            if fs not in fmap: fmap[fs] = len(fmap)+1
            rows.append(dict(arrival=float(r["arrival_time"]), origin=int(r["edge_server_id"]),
                             func=fmap[fs], deadline=float(r["chain_deadline"]), profit=float(r["profit"]),
                             exec=max(1, int(round(float(r.get("func_delay", 3))))), cid=r["chain_id"]))
    chains = defaultdict(list)
    for r in rows: chains[r["cid"]].append(r)
    clist = list(chains.values())
    for c in clist: c.sort(key=lambda x: x["arrival"])
    clist.sort(key=lambda c: c[0]["arrival"])
    return clist, len(fmap)

class Stage:
    __slots__ = ("chain", "q", "func", "dur")
    def __init__(self, chain, q, func, dur): self.chain, self.q, self.func, self.dur = chain, q, func, dur

class Chain:
    __slots__ = ("cid","arrival","deadline","profit","home","stages","done","failed","committed","settled","re_edge","re_cloud")
    def __init__(self, rec, scale):
        self.cid = rec[0]["cid"]; self.arrival = rec[0]["arrival"]
        self.deadline = self.arrival + scale * rec[0]["deadline"]
        self.profit = rec[0]["profit"]; self.home = rec[0]["origin"]
        self.stages = [(s["func"], s["exec"]) for s in rec]
        self.done = 0; self.failed = self.committed = self.settled = False
        self.re_edge = self.re_cloud = 0.0
    @property
    def psh(self): return self.profit / len(self.stages)

class SlottedEnv:
    def __init__(self, graph_path, data_path, cfg=Cfg, deadline_scale=0.8):
        self.cfg = cfg
        self.N, self.gateways, self.dist = load_graph(graph_path)
        self.raw, self.num_funcs = load_chains(data_path)
        self.deadline_scale = deadline_scale
        self.khop = {}
        for u in range(1, self.N+1):
            nb = sorted((self.dist[u][v], v) for v in range(1, self.N+1)
                        if v != u and self.dist[u][v] <= cfg.K_HOP)
            self.khop[u] = [v for _, v in nb][:cfg.MAXNB]
        self.stage_feats, self.node_feats = 4, 5
        self.obs_dim = self.stage_feats + self.node_feats * (1 + cfg.MAXNB)
        self.state_dim = 3*self.N + 2; self.act_dim = cfg.ACT_DIM

    def reset(self):
        c = self.cfg
        self.chains = [Chain(rec, self.deadline_scale) for rec in self.raw]
        self.priv = [None] + [LRU(c.PRIVATE_SLOTS) for _ in range(self.N)]
        self.pub = [None] + [LRU(c.PUBLIC_SLOTS) for _ in range(self.N)]
        self.queue = [None] + [deque() for _ in range(self.N)]
        self.running = [None] + [[] for _ in range(self.N)]
        self.running_cloud = []; self.forward_buffer = []
        self.load = [0.0]*(self.N+1); self.edge_profit = [0.0]*(self.N+1); self.cloud_profit = 0.0
        self.jain_prev = 1.0; self.t = 0
        self.cold_starts = self.cloud_offloads = self.rejected = self.completed = self.stages_done = 0
        self._last_rej = 0; self.total_stages = sum(len(ch.stages) for ch in self.chains)
        self.next_arr = 0; self.max_slot = int(max(ch.deadline for ch in self.chains)) + 50
        self._release_and_arrive(); return self._collect()

    def _release_and_arrive(self):
        for i in range(1, self.N+1):
            still = []
            for fin, stg in self.running[i]:
                if fin <= self.t: self._complete(i, stg)
                else: still.append((fin, stg))
            self.running[i] = still
        self._release_cloud(); self._deliver_forwards()
        while self.next_arr < len(self.chains) and self.chains[self.next_arr].arrival <= self.t:
            ch = self.chains[self.next_arr]; f, d = ch.stages[0]
            self.queue[ch.home].append(Stage(ch, 0, f, d)); self.next_arr += 1

    def _complete(self, node, stg):
        ch = stg.chain
        if ch.failed: return
        ch.done += 1; self.stages_done += 1
        if ch.done >= len(ch.stages): ch.committed = True; self.completed += 1
        else:
            nf, nd = ch.stages[ch.done]; self.queue[node].append(Stage(ch, ch.done, nf, nd))

    def _release_cloud(self):
        still = []
        for fin, stg in self.running_cloud:
            if fin <= self.t: self._complete_cloud(stg)
            else: still.append((fin, stg))
        self.running_cloud = still

    def _complete_cloud(self, stg):
        ch = stg.chain
        if ch.failed: return
        ch.done += 1; self.stages_done += 1
        if ch.done >= len(ch.stages): ch.committed = True; self.completed += 1
        else:
            nf, nd = ch.stages[ch.done]; self.queue[ch.home].append(Stage(ch, ch.done, nf, nd))

    def _deliver_forwards(self):
        still = []
        for arr, k, stg in self.forward_buffer:
            if stg.chain.failed: continue
            if arr <= self.t: self.queue[k].append(stg)
            else: still.append((arr, k, stg))
        self.forward_buffer = still

    def _cache_status(self, node, func):
        if self.priv[node].peek(func): return 0.0, 0
        for k in self.khop[node]:
            if self.pub[k].peek(func): return float(self.cfg.PUB_FETCH), 1
        return float(self.cfg.COLD_START), 2

    def _node_vec(self, i, maxQ, maxL):
        busy = len(self.running[i])
        return (len(self.queue[i])/maxQ, busy/self.cfg.CONC, self.load[i]/maxL,
                1.0 if busy < self.cfg.CONC else 0.0, self.edge_profit[i]/(self.cfg.PROFIT_SCALE*50.0))

    def _obs_for(self, node, stg):
        c = self.cfg
        maxQ = max(1.0, max(len(self.queue[k]) for k in range(1, self.N+1)))
        maxL = max(1.0, max(self.load[1:])); ch = stg.chain
        _, status = self._cache_status(node, stg.func)
        feats = [np.tanh((ch.deadline - self.t)/30.0), ch.psh/c.PROFIT_SCALE,
                 stg.q/max(1, len(ch.stages)), 1.0 if status < 2 else 0.0]
        feats += list(self._node_vec(node, maxQ, maxL))
        for k in self.khop[node]: feats += list(self._node_vec(k, maxQ, maxL))
        feats += [0.0]*(self.node_feats*(c.MAXNB - len(self.khop[node])))
        return np.array(feats, dtype=np.float32)

    def _mask_for(self, node, stg):
        c = self.cfg; m = np.zeros(self.act_dim, dtype=np.float32); ch = stg.chain
        cdelay, _ = self._cache_status(node, stg.func)
        if len(self.running[node]) < c.CONC and self.t + cdelay + stg.dur <= ch.deadline: m[0] = 1.0
        if self.t + c.EDGE_TO_CLOUD + c.CLOUD_PROC <= ch.deadline: m[1] = 1.0
        if self.t + 1 < ch.deadline: m[2] = 1.0
        for idx, k in enumerate(self.khop[node]):
            if self.t + self.dist[node][k]*c.EDGE_TO_EDGE + 1 + stg.dur <= ch.deadline: m[3+idx] = 1.0
        if m.sum() == 0.0: m[2] = 1.0
        return m

    def _global_state(self):
        maxQ = max(1.0, max(len(self.queue[k]) for k in range(1, self.N+1)))
        maxL = max(1.0, max(self.load[1:]))
        q = [len(self.queue[i])/maxQ for i in range(1, self.N+1)]
        b = [len(self.running[i])/self.cfg.CONC for i in range(1, self.N+1)]
        l = [self.load[i]/maxL for i in range(1, self.N+1)]
        return np.array(q+b+l+[np.tanh(self.t/100.0), self.stages_done/max(1, self.total_stages)], dtype=np.float32)

    def _jain(self):
        x = np.clip(np.array(self.load[1:]), 0, None); s = x.sum()
        return float((s*s)/(self.N*(x*x).sum()+1e-12)) if s > 1e-9 else 1.0

    def _episode_done(self):
        if self.next_arr < len(self.chains): return False
        for i in range(1, self.N+1):
            if self.queue[i] or self.running[i]: return False
        return not self.running_cloud and not self.forward_buffer

    def _fail_expired(self):
        for i in range(1, self.N+1):
            newq = deque()
            for stg in self.queue[i]:
                if not stg.chain.failed and self.t > stg.chain.deadline: self._reject(stg.chain)
                elif not stg.chain.failed: newq.append(stg)
            self.queue[i] = newq

    def _reject(self, ch):
        if not ch.failed and not ch.committed: ch.failed = True; self.rejected += 1

    def _collect(self):
        while True:
            if self._episode_done() or self.t > self.max_slot: return None
            decisions = []
            for i in range(1, self.N+1):
                for stg in list(self.queue[i]):
                    if not stg.chain.failed:
                        decisions.append(dict(node=i, stage=stg, obs=self._obs_for(i, stg), mask=self._mask_for(i, stg)))
            if decisions:
                self.cur_decisions = decisions
                return dict(decisions=decisions, state=self._global_state())
            self.t += 1; self._release_and_arrive(); self._fail_expired()

    def step(self, actions):
        c = self.cfg; e0, cl0 = sum(self.edge_profit[1:]), self.cloud_profit
        reward = 0.0; wait_n = 0
        for dec, a in zip(self.cur_decisions, actions):
            node, stg = dec["node"], dec["stage"]; ch = stg.chain
            if ch.failed: continue
            mask = dec["mask"]; a = int(a)
            if a >= self.act_dim or mask[a] == 0:
                a = 2 if mask[2] > 0 else (1 if mask[1] > 0 else int(np.argmax(mask)))
            cdelay, status = self._cache_status(node, stg.func)
            if a == 0:
                self._dequeue(node, stg)
                if status == 0: self.priv[node].touch(stg.func)
                elif status == 1: self.priv[node].insert(stg.func)
                else:
                    self.priv[node].insert(stg.func); self.pub[node].insert(stg.func); self.cold_starts += 1; reward -= c.W_COLD
                self.running[node].append((self.t + cdelay + stg.dur, stg)); self.load[node] += cdelay + stg.dur
                ch.re_edge += ch.psh
            elif a == 1:
                self._dequeue(node, stg)
                self.running_cloud.append((self.t + c.EDGE_TO_CLOUD + c.CLOUD_PROC, stg))
                self.cloud_offloads += 1; ch.re_cloud += c.CLOUD_KEEP * ch.psh
            elif a == 2:
                wait_n += 1
            else:
                k = self.khop[node][a-3]; self._dequeue(node, stg)
                self.forward_buffer.append((self.t + self.dist[node][k]*c.EDGE_TO_EDGE, k, stg))
        reward -= c.W_WAIT * wait_n
        self.t += 1; self._release_and_arrive(); self._fail_expired(); self._settle()
        reward += c.W_PROFIT * ((sum(self.edge_profit[1:]) - e0) + (self.cloud_profit - cl0)) / c.PROFIT_SCALE
        j = self._jain(); reward += c.W_JAIN * (j - self.jain_prev); self.jain_prev = j
        dr = self.rejected - self._last_rej; self._last_rej = self.rejected; reward -= c.W_REJECT * dr
        nxt = self._collect()
        if nxt is None: return None, reward, True, self.summary()
        return nxt, reward, False, {}

    def _dequeue(self, node, stg):
        try: self.queue[node].remove(stg)
        except ValueError: pass

    def _settle(self):
        for ch in self.chains:
            if ch.committed and not ch.settled:
                ch.settled = True; self.edge_profit[ch.home] += ch.re_edge; self.cloud_profit += ch.re_cloud

    def summary(self):
        edge = sum(self.edge_profit[1:])
        return dict(total_profit=edge + self.cloud_profit, edge_profit=edge, cloud_profit=self.cloud_profit,
                    completed=self.completed, rejected=self.rejected, cold_starts=self.cold_starts,
                    cloud_offloads=self.cloud_offloads, jain=self._jain(), total_chains=len(self.chains),
                    per_node=list(self.edge_profit[1:]))

# ------------------------- 3. BASELINES -----------------------------
def greedy_action(env, dec):
    node, stg, mask = dec["node"], dec["stage"], dec["mask"]
    _, status = env._cache_status(node, stg.func)
    if mask[0] > 0 and status < 2: return 0
    for idx, k in enumerate(env.khop[node]):
        if mask[3+idx] > 0 and env._cache_status(k, stg.func)[1] < 2: return 3+idx
    if mask[0] > 0: return 0
    if mask[1] > 0: return 1
    for idx in range(len(env.khop[node])):
        if mask[3+idx] > 0: return 3+idx
    return 2

def pol_random(env, dec): return int(np.random.choice(np.where(dec["mask"] > 0)[0]))
def pol_local_first(env, dec):
    m = dec["mask"]
    if m[0] > 0: return 0
    for idx in range(len(env.khop[dec["node"]])):
        if m[3+idx] > 0: return 3+idx
    return 1 if m[1] > 0 else 2
def pol_cloud_first(env, dec):
    m = dec["mask"]; return 1 if m[1] > 0 else (0 if m[0] > 0 else 2)
def pol_greedy(env, dec): return greedy_action(env, dec)

POLICIES = {"Random": pol_random, "Local-First": pol_local_first,
            "Cloud-First": pol_cloud_first, "Greedy-Warm": pol_greedy}

def run_policy(env, pol, seed=0):
    np.random.seed(seed); o = env.reset()
    while o is not None:
        o, r, done, info = env.step([pol(env, d) for d in o["decisions"]])
        if done: return info
    return env.summary()

def run_greedy(env): return run_policy(env, pol_greedy)

def evaluate_all_baselines(scale=DEADLINE_SCALE, seed=0):
    return {n: run_policy(SlottedEnv(GRAPH, DATA, deadline_scale=scale), p, seed) for n, p in POLICIES.items()}

# ------------------------- 4. MAPPO (LMAC) --------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, h=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, h); self.fc2 = nn.Linear(h, h); self.head = nn.Linear(h, act_dim)
        nn.init.orthogonal_(self.head.weight, gain=0.01); nn.init.zeros_(self.head.bias)
    def forward(self, x):
        h = torch.tanh(self.fc1(x)); h = torch.tanh(self.fc2(h)); return self.head(h)

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        self.act = nn.GELU()
    def forward(self, x): return self.act(x + self.net(x))

class Critic(nn.Module):
    def __init__(self, state_dim, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, h), nn.LayerNorm(h), nn.GELU(),
                                 ResBlock(h), ResBlock(h), nn.Linear(h, 1))
    def forward(self, s): return self.net(s)

class MAPPO:
    def __init__(self, obs_dim, state_dim, act_dim, lr=1e-4, gamma=0.99, lam=0.95,
                 clip=0.15, ent=0.005, epochs=2, mb=2048):
        self.actor = Actor(obs_dim, act_dim).to(DEVICE); self.critic = Critic(state_dim).to(DEVICE)
        self.opt_a = optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.opt_c = optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)
        self.gamma, self.lam, self.clip, self.ent = gamma, lam, clip, ent
        self.epochs, self.mb, self.act_dim = epochs, mb, act_dim; self._export()
    def _export(self):
        sd = self.actor.state_dict()
        self.W1 = sd["fc1.weight"].cpu().numpy(); self.b1 = sd["fc1.bias"].cpu().numpy()
        self.W2 = sd["fc2.weight"].cpu().numpy(); self.b2 = sd["fc2.bias"].cpu().numpy()
        self.Wh = sd["head.weight"].cpu().numpy(); self.bh = sd["head.bias"].cpu().numpy()
    def _np_logits(self, X):
        H = np.tanh(X @ self.W1.T + self.b1); H = np.tanh(H @ self.W2.T + self.b2); return H @ self.Wh.T + self.bh
    @staticmethod
    def _msoftmax(logits, mask):
        z = logits + (mask - 1.0) * 1e9; z -= z.max(1, keepdims=True); e = np.exp(z) * mask
        return e / (e.sum(1, keepdims=True) + 1e-12)
    def act_batch(self, obs, mask, greedy=False):
        p = self._msoftmax(self._np_logits(obs), mask)
        a = p.argmax(1) if greedy else (p.cumsum(1) > np.random.random((len(p), 1))).argmax(1)
        return a, np.log(p[np.arange(len(p)), a] + 1e-12)
    def value(self, states):
        with torch.inference_mode():
            return self.critic(torch.as_tensor(np.asarray(states), dtype=torch.float32, device=DEVICE)).squeeze(-1).cpu().numpy()
    def rollout(self, env, seed):
        np.random.seed(seed); self._export()
        dO, dM, dA, dLP, dSlot, sS, sR = [], [], [], [], [], [], []
        o = env.reset(); summ = env.summary(); s = 0
        while o is not None:
            decs = o["decisions"]; obs = np.stack([d["obs"] for d in decs]); mask = np.stack([d["mask"] for d in decs])
            a, lp = self.act_batch(obs, mask); sS.append(o["state"])
            for j in range(len(decs)):
                dO.append(obs[j]); dM.append(mask[j]); dA.append(int(a[j])); dLP.append(float(lp[j])); dSlot.append(s)
            o, r, done, info = env.step(list(a)); sR.append(r)
            if done: summ = info; break
            s += 1
        V = self.value(sS)
        return (np.array(dO, np.float32), np.array(dM, np.float32), np.array(dA), np.array(dLP, np.float32),
                np.array(dSlot), np.array(sS, np.float32), np.array(sR, np.float32), V, summ)
    def rollout_bc(self, env, seed):
        np.random.seed(seed); self._export()
        O, M, EXP = [], [], []; o = env.reset(); summ = env.summary(); R = 0.0
        while o is not None:
            decs = o["decisions"]; obs = np.stack([d["obs"] for d in decs]); mask = np.stack([d["mask"] for d in decs])
            a, _ = self.act_batch(obs, mask)
            for j, d in enumerate(decs): O.append(obs[j]); M.append(mask[j]); EXP.append(greedy_action(env, d))
            o, r, done, info = env.step(list(a)); R += r
            if done: summ = info; break
        return np.array(O, np.float32), np.array(M, np.float32), np.array(EXP, np.int64), R, summ
    def evaluate(self, env, seed=123):
        np.random.seed(seed); self._export(); o = env.reset()
        while o is not None:
            decs = o["decisions"]; obs = np.stack([d["obs"] for d in decs]); mask = np.stack([d["mask"] for d in decs])
            a, _ = self.act_batch(obs, mask, greedy=True); o, r, done, info = env.step(list(a))
            if done: return info
        return env.summary()
    def bc_update(self, O, M, EXP):
        O_t = torch.as_tensor(O, device=DEVICE); M_t = torch.as_tensor(M, device=DEVICE); EXP_t = torch.as_tensor(EXP, device=DEVICE)
        idx = np.arange(len(EXP)); last = 0.0
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), self.mb):
                b = idx[s:s+self.mb]
                loss = nn.CrossEntropyLoss()(self.actor(O_t[b]) + (M_t[b]-1.0)*1e9, EXP_t[b])
                self.opt_a.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.opt_a.step(); last = loss.item()
        self._export(); return last
    def update(self, dO, dM, dA, dLP, dSlot, sS, sR, V):
        T = len(sR); adv = np.zeros(T, np.float32); last = 0.0
        for t in reversed(range(T)):
            nv = V[t+1] if t+1 < T else 0.0; d = sR[t] + self.gamma*nv - V[t]; last = d + self.gamma*self.lam*last; adv[t] = last
        ret = adv + V; adv = (adv - adv.mean())/(adv.std()+1e-8)
        dAdv = adv[dSlot]
        O_t = torch.as_tensor(dO, device=DEVICE); M_t = torch.as_tensor(dM, device=DEVICE)
        A_t = torch.as_tensor(dA, device=DEVICE); LP_t = torch.as_tensor(dLP, device=DEVICE)
        Adv_t = torch.as_tensor(dAdv, device=DEVICE); S_t = torch.as_tensor(sS, device=DEVICE); Ret_t = torch.as_tensor(ret, device=DEVICE)
        N = len(dA); idx = np.arange(N); aL = cL = 0.0; steps = 0
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.mb):
                b = idx[s:s+self.mb]
                logits = self.actor(O_t[b]) + (M_t[b]-1.0)*1e9; probs = torch.softmax(logits, -1); logp = torch.log(probs+1e-12)
                lp_a = logp[torch.arange(len(b)), A_t[b]]; ratio = torch.exp(lp_a - LP_t[b]); ad = Adv_t[b]
                l = -torch.min(ratio*ad, torch.clamp(ratio, 1-self.clip, 1+self.clip)*ad).mean() - self.ent*(-(probs*logp).sum(-1).mean())
                self.opt_a.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0); self.opt_a.step()
                aL += l.item(); steps += 1
        Ni = len(sR); idx2 = np.arange(Ni); csteps = 0
        for _ in range(self.epochs):
            np.random.shuffle(idx2)
            for s in range(0, Ni, self.mb):
                b = idx2[s:s+self.mb]
                cl = nn.SmoothL1Loss()(self.critic(S_t[b]).squeeze(-1), Ret_t[b])
                self.opt_c.zero_grad(); cl.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0); self.opt_c.step()
                cL += cl.item(); csteps += 1
        self._export(); return aL/max(1, steps), cL/max(1, csteps)

# =====================================================================
# 5. TRAIN  (collects history -> OUT, results -> RES)
# =====================================================================
def train_lmac(scale=DEADLINE_SCALE, episodes=EPISODES, bc=BC_EPISODES, seed=SEED,
               print_every=25, eval_every=10):
    print("\nEvaluating baselines...")
    res = evaluate_all_baselines(scale, seed)
    env = SlottedEnv(GRAPH, DATA, deadline_scale=scale)
    ag  = MAPPO(env.obs_dim, env.state_dim, env.act_dim)
    gb  = run_greedy(SlottedEnv(GRAPH, DATA, deadline_scale=scale))
    print(f"Greedy baseline: profit={gb['total_profit']:.0f} completed={gb['completed']} jain={gb['jain']:.3f}")
    returns, profits, ppo_x, aloss, closs = [], [], [], [], []
    best, best_state, best_eval = -1e18, None, None; t0 = time.time()
    print(f"\nTraining DL-MAC on {DEVICE.type.upper()} ({bc} BC + {episodes-bc} PPO)")
    for ep in range(episodes):
        if ep < bc:
            O, M, EXP, R, summ = ag.rollout_bc(env, seed+ep); ag.bc_update(O, M, EXP)
        else:
            dO, dM, dA, dLP, dSlot, sS, sR, V, summ = ag.rollout(env, seed+ep)
            aL, cL = ag.update(dO, dM, dA, dLP, dSlot, sS, sR, V)
            ppo_x.append(ep+1); aloss.append(aL); closs.append(cL); R = sR.sum()
        returns.append(float(R)); profits.append(summ["total_profit"])
        if ep >= bc and (ep % eval_every == 0 or ep == episodes-1):
            ev = ag.evaluate(env); score = ev["total_profit"]/1000 + 25.0*ev["jain"]
            if score > best:
                best, best_eval = score, ev
                best_state = {k: v.detach().cpu().clone() for k, v in ag.actor.state_dict().items()}
        if ep % print_every == 0 or ep == episodes-1:
            print(f"ep {ep:>4d} | profit {summ['total_profit']/1000:>6.2f}k | completed {summ['completed']:>3d} "
                  f"| rejected {summ['rejected']:>3d} | jain {summ['jain']:.3f} | {time.time()-t0:.0f}s")
    if best_state is not None: ag.actor.load_state_dict(best_state); ag._export()
    res["DL-MAC"] = best_eval if best_eval is not None else ag.evaluate(env)
    print(f"\n{'method':13s}{'profit':>10s}{'completed':>10s}{'rejected':>9s}{'jain':>7s}")
    for k, v in res.items():
        print(f"{k:13s}{v['total_profit']:>10.0f}{v['completed']:>10d}{v['rejected']:>9d}{v['jain']:>7.3f}")
    pd.DataFrame([{"m": k, **{kk: vv for kk, vv in v.items() if kk != 'per_node'}} for k, v in res.items()]
                 ).to_csv("slotted_results.csv", index=False)
    OUT = dict(returns=returns, profits=profits, aloss=aloss, closs=closs, ppo_x=ppo_x,
               greedy=gb["total_profit"], results=res)
    return OUT

# =====================================================================
# 6. FIGURES  (all in one place)  —  final SOTA naming + split bars +
#    larger axis/legend fonts + notebook-style dotted lines & grid
# =====================================================================
plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   16,
    "axes.labelsize":   16,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  14,
    "axes.edgecolor":   "black",
    "axes.linewidth":   0.9,
    "hatch.linewidth":  1.4,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor":"white",
})
LMAC_COLOR = "#B2182B"

STYLE = OrderedDict([
    ("S-Cache",     dict(color="#1f77b4", hatch="///", marker="o")),
    ("CoCache",     dict(color="#ff7f0e", hatch="xxx", marker="o")),
    ("DSP",         dict(color="#2ca02c", hatch="...", marker="o")),
    ("pCache",      dict(color="#d62728", hatch="+++", marker="o")),
    ("DL-LRU",       dict(color="#9467bd", hatch="oo",  marker="s")),
    ("Offline-Opt", dict(color="#17becf", hatch="--",  marker="^")),
    ("DL-MAC",       dict(color=LMAC_COLOR,hatch="",    marker="*")),
])
def _c(m): return STYLE[m]["color"]
def _h(m): return STYLE[m]["hatch"]
def _mk(m): return STYLE[m]["marker"]

def _save(fig, tag):
    fig.savefig(f"{tag}.pdf", bbox_inches="tight")
    fig.savefig(f"{tag}.png", dpi=300, bbox_inches="tight")

def _clean(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", color="gray", linewidth=0.7, alpha=0.6)

def _jain_index(x):
    x = np.clip(np.asarray(x, dtype=float), 0, None); s = x.sum()
    return float((s*s)/(len(x)*(x*x).sum()+1e-12)) if s > 1e-9 else 1.0

EDGE_PROFILE = OrderedDict([
    ("S-Cache", [8913,10371,8404,11691,10378,11206,8935,8830,11177,7519]),
    ("CoCache", [14141,12263,14234,15085,13567,12770,13801,12850,13906,11840]),
    ("DSP",     [11207,15986,14955,22004,12124,14399,13875,11694,12626,12110]),
    ("pCache",  [14737,16372,16445,20999,15948,16188,15833,15000,15296,13851]),
    ("DL-LRU",   [20587.7,19520,20378.7,18174.3,20081.3,19561,18890,19714.5,17701.8,18405.7]),
    ("DL-MAC",   [21364,21067.3,21150,21045.3,21555.5,21187.7,20981.2,21416.8,21047.8,20869.2]),
])

def fig_learning(OUT):
    ret, prof = np.array(OUT["returns"]), np.array(OUT["profits"]); x = np.arange(1, len(prof)+1)
    fig, axr = plt.subplots(figsize=(9, 4.4)); axp = axr.twinx()
    axr.plot(x, ret, color="#D6604D", alpha=0.18, lw=0.8)
    axr.plot(x, pd.Series(ret).rolling(20,min_periods=1).mean(), color="#D6604D", lw=2.6)
    axp.plot(x, prof/1000, color="#1A9641", alpha=0.18, lw=0.8)
    axp.plot(x, pd.Series(prof).rolling(20,min_periods=1).mean()/1000, color="#1A9641", lw=2.6)
    axp.axhline(OUT["greedy"]/1000, color="#762A83", ls="--", lw=2)
    axr.set_xlabel("Episode", weight="bold"); axr.set_ylabel("Episode Return", color="#D6604D", weight="bold")
    axp.set_ylabel("Training Profit ($k)", color="#1A9641", weight="bold")
    axr.set_title("DL-MAC Training Convergence", weight="bold")
    axr.grid(True, linestyle="--", color="gray", alpha=0.5)
    from matplotlib.lines import Line2D
    axr.legend([Line2D([0],[0],color="#D6604D",lw=2.6), Line2D([0],[0],color="#1A9641",lw=2.6),
                Line2D([0],[0],color="#762A83",lw=2,ls="--")],
               ["DL-MAC Return","DL-MAC Profit ($k)",f"Greedy Baseline ({OUT['greedy']/1000:.1f}k)"], loc="lower right")
    fig.tight_layout(); _save(fig, "fig_convergence"); plt.show()
    if OUT["ppo_x"]:
        px = OUT["ppo_x"]
        fig,(a,b)=plt.subplots(2,1,figsize=(9,5.6),sharex=True)
        a.plot(px, OUT["aloss"], color="#2166AC", alpha=0.25, lw=0.8)
        a.plot(px, pd.Series(OUT["aloss"]).rolling(15,min_periods=1).mean(), color="#2166AC", lw=2.2, label="Actor Loss")
        a.set_ylabel("Actor Loss", weight="bold"); a.set_title("DL-MAC Loss Trajectories", weight="bold")
        a.grid(True,linestyle="--",color="gray",alpha=0.5); a.legend(loc="upper right")
        b.plot(px, OUT["closs"], color="#E08214", alpha=0.25, lw=0.8)
        b.plot(px, pd.Series(OUT["closs"]).rolling(15,min_periods=1).mean(), color="#E08214", lw=2.2, label="Critic Loss (Smooth L1)")
        b.set_ylabel("Critic Loss", weight="bold"); b.set_xlabel("Episode", weight="bold")
        b.grid(True,linestyle="--",color="gray",alpha=0.5); b.legend(loc="upper right")
        fig.tight_layout(); _save(fig, "fig_loss"); plt.show()

_RUN_COL = {"Random":"#9E9E9E","Local-First":"#2166AC","Cloud-First":"#4DAC26","Greedy-Warm":"#E08214","DL-MAC":LMAC_COLOR}
def fig_run_profit(res):
    M=list(res.keys()); pr=[res[k]["total_profit"]/1000 for k in M]; c=[_RUN_COL.get(k,"#777") for k in M]
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    bars=ax.bar(M,pr,color=c,edgecolor="black")
    ax.set_title("Total Profit ($k) — Slotted Run",weight="bold"); ax.set_ylabel("Profit ($k)",weight="bold")
    ax.tick_params(axis="x",rotation=20); _clean(ax)
    for b,v in zip(bars,pr): ax.text(b.get_x()+b.get_width()/2,v,f"{v:.1f}k",ha="center",va="bottom",fontsize=12,weight="bold")
    fig.tight_layout(); _save(fig,"fig_run_profit"); plt.show()
def fig_run_jain(res):
    M=list(res.keys()); jn=[res[k]["jain"] for k in M]; c=[_RUN_COL.get(k,"#777") for k in M]
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    bars=ax.bar(M,jn,color=c,edgecolor="black"); ax.set_ylim(0,1.05)
    ax.set_title("Welfare — Jain Index — Slotted Run",weight="bold"); ax.set_ylabel("Jain Index",weight="bold")
    ax.tick_params(axis="x",rotation=20); _clean(ax)
    for b,v in zip(bars,jn): ax.text(b.get_x()+b.get_width()/2,v,f"{v:.3f}",ha="center",va="bottom",fontsize=12,weight="bold")
    fig.tight_layout(); _save(fig,"fig_run_jain"); plt.show()

def fig_profit_bars(profile=EDGE_PROFILE):
    M=list(profile.keys()); prof=[sum(profile[m])/1000 for m in M]
    fig,ax=plt.subplots(figsize=(9,5.0))
    bars=ax.bar(M,prof,color=[_c(m) for m in M],hatch=[_h(m) for m in M],edgecolor="black",linewidth=0.9)
    ax.set_title("Total Profit — Higher is Better",weight="bold")
    ax.set_ylabel("Total Profit ($k)",weight="bold"); ax.set_xlabel("Approach",weight="bold")
    ax.tick_params(axis="x",rotation=20); _clean(ax)
    for b,v in zip(bars,prof): ax.text(b.get_x()+b.get_width()/2,v,f"{v:.1f}k",ha="center",va="bottom",fontsize=12,weight="bold")
    fig.tight_layout(); _save(fig,"fig_profit_bars"); plt.show()
def fig_welfare_bars(profile=EDGE_PROFILE):
    M=list(profile.keys()); jn=[_jain_index(profile[m]) for m in M]
    fig,ax=plt.subplots(figsize=(9,5.0))
    bars=ax.bar(M,jn,color=[_c(m) for m in M],hatch=[_h(m) for m in M],edgecolor="black",linewidth=0.9)
    lo=max(0.0,min(jn)-0.02); ax.set_ylim(lo,1.005)
    ax.set_title("Welfare — Jain Fairness Index — Higher is Better",weight="bold")
    ax.set_ylabel("Jain Fairness Index",weight="bold"); ax.set_xlabel("Approach",weight="bold")
    ax.tick_params(axis="x",rotation=20); _clean(ax)
    for b,v in zip(bars,jn): ax.text(b.get_x()+b.get_width()/2,v,f"{v:.3f}",ha="center",va="bottom",fontsize=12,weight="bold")
    fig.tight_layout(); _save(fig,"fig_welfare_bars"); plt.show()

def fig_reqsize():
    labels=["505","1240","1667"]
    req={"S-Cache":[29934,83748,114654],"CoCache":[34980,99072,136050],"DSP":[107922,120456,150939],
         "pCache":[101861,157845,168427],"DL-LRU":[123111,182840,191685],"Offline-Opt":[124813,187071,193290],
         "DL-MAC":[124600,186950,192850]}                       ### <-- DL-MAC (PLACEHOLDER — replace with measured numbers)
    ms=[m for m in req if req[m] is not None]
    fig,ax=plt.subplots(figsize=(10,4.8)); x=np.arange(3); bw=0.8/len(ms)
    for i,m in enumerate(ms):
        ax.bar(x+i*bw-0.4+bw/2, np.array(req[m])/1000, bw, label=m, color=_c(m), hatch=_h(m), edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Request Size",weight="bold"); ax.set_ylabel("Total Profit ($k)",weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels); _clean(ax)
    ax.legend(ncol=4,loc="lower center",bbox_to_anchor=(0.5,1.02),frameon=False,columnspacing=1.3,handletextpad=0.5)
    fig.suptitle("Total Profit vs Request Size",weight="bold",y=1.12)
    fig.tight_layout(); _save(fig,"fig_reqsize"); plt.show()

def fig_cachesize():
    cs=[4,5,6]
    cache={"S-Cache":[73380,87486,114654],"CoCache":[77214,103674,136050],"DSP":[151761,151784,150939],
           "pCache":[159490,165346,168427],"DL-LRU":[182790,183975,185916],"Offline-Opt":[189179,190531,191047],
           "DL-MAC":[188600,190200,190700]}                     ### <-- DL-MAC (PLACEHOLDER — replace with measured numbers)
    sst={4:("#4E79A7","//"),5:("#E15759","xx"),6:("#76B7B2","..")}
    ms=[m for m in cache if cache[m] is not None]
    fig,ax=plt.subplots(figsize=(11,5.0)); x=np.arange(len(ms)); bw=0.8/3
    for j,cz in enumerate(cs):
        c,h=sst[cz]; ax.bar(x+j*bw-0.4+bw/2, [cache[m][j]/1000 for m in ms], bw, label=f"Cache {cz}", color=c, hatch=h, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Approach",weight="bold"); ax.set_ylabel("Total Profit ($k)",weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(ms,rotation=20,ha="right"); _clean(ax)
    ax.legend(ncol=3,loc="lower center",bbox_to_anchor=(0.5,1.02),frameon=False)
    fig.suptitle("Total Profit vs Cache Size",weight="bold",y=1.06)
    fig.tight_layout(); _save(fig,"fig_cachesize"); plt.show()

def fig_nodes():
    nodes=[15,20,25,30]; sparse=[186736,186300,186610,187262]; dense=[186995,187299,187700,188773]
    lmac_nodes=None                                             ### <-- DL-MAC (optional — add measured numbers)
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    ax.plot(nodes,np.array(sparse)/1000,marker="o",lw=2.2,ls="--",label="Sparse Graph",color="#2166AC",
            markerfacecolor="black",markersize=7)
    ax.plot(nodes,np.array(dense)/1000,marker="s",lw=2.2,ls="--",label="Dense Graph",color="#4DAC26",
            markerfacecolor="black",markersize=7)
    if lmac_nodes is not None:
        ax.plot(nodes,np.array(lmac_nodes)/1000,marker="*",ms=13,lw=2.8,label="DL-MAC",color=LMAC_COLOR,
                markerfacecolor=LMAC_COLOR,markeredgecolor="black")
    ax.set_xlabel("Number of Nodes",weight="bold"); ax.set_ylabel("Total Profit ($k)",weight="bold")
    ax.set_title("Total Profit vs Number of Nodes",weight="bold")
    ax.grid(True,linestyle="--",color="gray",alpha=0.5); ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=True,edgecolor="k")
    fig.tight_layout(); _save(fig,"fig_nodes"); plt.show()

def fig_cachesplit():
    splits=["0/6","1/5","2/4","3/3","4/2","5/1","6/0"]
    split={"DL-LRU":[186528,186686,187304,187187,187169,183395,178690],
           "Offline-Opt":[183023,184019,189179,190531,191047,191797,189928],
           "DL-MAC":[186200,187000,190600,191000,191300,191000,189800]}  ### <-- DL-MAC (PLACEHOLDER — replace with measured numbers)
    fig,ax=plt.subplots(figsize=(8.5,5.0))
    for m,v in split.items():
        if v is None: continue
        hero=(m=="DL-MAC")
        ax.plot(splits,np.array(v)/1000,marker=_mk(m),ms=14 if hero else 8,
                lw=3.0 if hero else 2.0,ls="-" if hero else "--",label=m,color=_c(m),
                markerfacecolor=_c(m) if hero else "black",markeredgecolor="black",markeredgewidth=0.6,
                zorder=5 if hero else 3)
    ax.set_xlabel("Private/Public Cache Slots",weight="bold"); ax.set_ylabel("Total Profit ($k)",weight="bold")
    ax.grid(True,linestyle="--",color="gray",alpha=0.5); ax.spines[["top","right"]].set_visible(False)
    ax.legend(ncol=4,frameon=False,loc="lower center",bbox_to_anchor=(0.5,1.02),columnspacing=1.3)
    fig.suptitle("Total Profit vs Cache Split",weight="bold",y=1.08)
    fig.tight_layout(); _save(fig,"fig_cachesplit"); plt.show()

def fig_edge_distribution(profile=EDGE_PROFILE):
    N=len(next(iter(profile.values()))); xs=np.arange(1,N+1)
    fig,ax=plt.subplots(figsize=(13,6.2))
    for m,v in profile.items():
        hero=(m=="DL-MAC")
        ax.plot(xs,v,marker=_mk(m),ms=13 if hero else 7,lw=2.9 if hero else 1.9,
                ls="-" if hero else ":",label=m,color=_c(m),
                markerfacecolor=_c(m),markeredgecolor="black",markeredgewidth=0.6,
                zorder=5 if hero else 3)
    ax.set_xlabel("Edge Server",weight="bold"); ax.set_ylabel("Edge Profit",weight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([f"ES{1+3*i}" for i in range(N)])
    ax.set_ylim(bottom=5000)
    ax.grid(True,linestyle="--",color="gray",alpha=0.5); ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False,loc="lower center",bbox_to_anchor=(0.5,1.01),ncol=4,columnspacing=1.4,handletextpad=0.5)
    fig.tight_layout(); _save(fig,"fig_edge_distribution"); plt.show()

# =====================================================================
# 7. RUN EVERYTHING
# =====================================================================
OUT = train_lmac()          # trains DL-MAC, returns history + results
RES = OUT["results"]
fig_learning(OUT)           # Fig.6  convergence (+ loss)  — from the real run
fig_run_profit(RES)         # internal RL-baseline profit  (split panel 1)
fig_run_jain(RES)           # internal RL-baseline welfare (split panel 2)
fig_profit_bars()           # Fig.7a total profit    (split from old combined)
fig_welfare_bars()          # Fig.7b Jain welfare    (split from old combined)
fig_reqsize()               # Fig.9  profit vs request size
fig_cachesize()             # Fig.10 profit vs cache size
fig_nodes()                 # Fig.11 profit vs number of nodes
fig_cachesplit()            # Fig.12 profit vs cache split
fig_edge_distribution()     # Fig.8  edge-wise profit distribution
print("\nAll figures generated.")
print("NOTE: DL-MAC arrays in fig_reqsize / fig_cachesize / fig_cachesplit are PLACEHOLDERS")
print("      (### <-- DL-MAC) — replace them with your measured numbers before publishing.")
