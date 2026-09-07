#!/usr/bin/env python3
"""
Experiment driver: run a workflow (chain) under a chosen scheduling policy on a
pool of edge-server task-runners, with MEASURED cold starts / compute / transport,
and write per-task and per-run CSVs.

Example:
  python run_dag_workflow.py --workflow log_processing --policy dl_lru --instances 4
  python run_dag_workflow.py --workflow video_analytics --policy dl_mac --instances 3

Backends: --backend local (subprocess task-runners; default, no Docker) or
point pool.py at Docker-compose/k8s pod URLs for a real cluster (see README).
"""
import os, csv, json, time, argparse, random, collections
from scheduler.pool import EdgePool
from scheduler.cost_model import Cfg, transport, jain
from scheduler import policy as P

HERE = os.path.dirname(os.path.abspath(__file__))

def load_graph(path):
    toks = open(path).read().split(); it = iter(toks)
    N, M, Gn = int(next(it)), int(next(it)), int(next(it))
    gws = [int(next(it)) for _ in range(Gn)]
    adj = collections.defaultdict(list)
    for _ in range(M):
        u, v = int(next(it)), int(next(it)); adj[u].append(v); adj[v].append(u)
    INF = 10**9; dist = {}
    for s in range(1, N + 1):
        d = {i: INF for i in range(1, N + 1)}; d[s] = 0; q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[v] > d[u] + 1: d[v] = d[u] + 1; q.append(v)
        d["gw"] = min(d[g] for g in gws)  # hops to nearest gateway
        dist[s] = d
    return N, gws, dist

def khop(dist, home, N, K):
    return [i for i in range(1, N + 1) if i != home and dist[home][i] <= K]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--policy", required=True, choices=list(P.POLICIES))
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--deadline-factor", type=float, default=1.4)
    ap.add_argument("--profit-per-stage", type=float, default=10.0)
    ap.add_argument("--backend", default="local")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); random.seed(a.seed)

    wf = json.load(open(os.path.join(HERE, "workflows_real", f"{a.workflow}.json")))
    stages = wf["stages"]; nS = len(stages)
    N, gws, dist = load_graph(os.path.join(HERE, "data", "graph.txt"))
    pool = EdgePool(os.path.join(HERE, "data", "_scratch"), backend=a.backend, cold_penalty=Cfg.cold)
    pick = P.POLICIES[a.policy]

    load = collections.defaultdict(float)     # per-node cumulative work (for Jain / price)
    demand = collections.defaultdict(lambda: collections.defaultdict(int))
    served = 0; total_profit = 0.0; cold = 0; task_rows = []
    t0 = time.time()
    for j in range(a.instances):
        home = random.randint(1, N); cand = [home] + khop(dist, home, N, a.K) + ["CC"]
        p_j = a.profit_per_stage * nS
        finish = {-1: 0.0}; chain_time = 0.0; ok = True
        for st in stages:                     # topological (ids are in dep order here)
            ftype = st["type"]
            warm = {n: ((n, ftype) in pool.warm) for n in cand if n != "CC"}
            ctx = dict(home=home, candidates=cand, warm=warm,
                       price={n: load[n] for n in cand}, load={n: load[n] for n in cand},
                       demand={n: demand[n][ftype] for n in cand if n != "CC"},
                       stage_work=1.0, profit=p_j / nS, model=None)
            node = pick(ctx)
            server_id = node if node != "CC" else "CC"
            r = pool.run(server_id, ftype, ftype, st["args"])
            tr = transport(home, node, dist)
            dep_done = max([finish[d] for d in st["deps"]], default=chain_time)
            fin = dep_done + tr + r["cold_s"] + r["compute_s"]
            finish[st["id"]] = fin; chain_time = max(chain_time, fin)
            work = r["cold_s"] + r["compute_s"]
            if node != "CC": load[node] += work; demand[node][ftype] += 1
            if r["cold_s"] > 0: cold += 1
            task_rows.append(dict(chain=j, stage=st["id"], type=ftype, node=node,
                                  cold_s=r["cold_s"], compute_s=r["compute_s"], transport_s=round(tr,3)))
        # deadline: factor x warm critical path (compute + transport) plus a
        # tolerance for a few cold starts; cold-heavy placements miss and earn 0.
        base = sum(t["compute_s"] + t["transport_s"] for t in task_rows if t["chain"] == j)
        deadline = a.deadline_factor * base + Cfg.cold * ((nS + 1) // 2)
        if chain_time <= deadline: served += 1; total_profit += p_j
    wall = time.time() - t0
    loads = [load[i] for i in range(1, N + 1)]
    summ = dict(workflow=a.workflow, policy=a.policy, instances=a.instances,
                served=served, rejected=a.instances - served, total_profit=round(total_profit, 1),
                cold_starts=cold, warm_hits=pool.warm_hits, jain=round(jain(loads), 4),
                wall_s=round(wall, 1))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tag = f"{a.workflow}_{a.policy}_n{a.instances}"
    with open(os.path.join(HERE, "results", tag + "_tasks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=task_rows[0].keys()); w.writeheader(); w.writerows(task_rows)
    with open(os.path.join(HERE, "results", tag + "_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summ.keys()); w.writeheader(); w.writerow(summ)
    pool.shutdown()
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
