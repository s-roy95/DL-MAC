
# DL-MAC Edge–Cloud FaaS — Deployment & Training

Companion code for *"Maximizing Total Profit and Welfare through Request
Scheduling and Caching in Edge–Cloud FaaS Systems."* Three parts:

1. **Docker implementation + datasets** — real containerized function chains.
2. **`lmac_dl.py`** — DL-MAC trainer on the real dataset.
3. **`run_dag_workflow.py`** — the experiment driver.

---

## 1. Docker implementation & datasets

Each function instance is a `task-runner` container (`python:3.11-slim`) running
one HTTP server; the scheduler POSTs a stage and measures the real cold start,
compute, and transport. Two deep chains ship: `staircase_chain` (42 stages) and
`random_chain` (40 stages), both CPU-bound `transform` chains.

```bash
# containerized edge instances (es1..es3 + gw, shared /data volume)
docker compose up --build
curl localhost:8081/ready
curl -s -X POST localhost:8081/run -d '{"stage":"transform","args":{"rounds":3,"kb":256}}'

# real cluster (kind / Kubernetes)
kind create cluster --name dlmac --config kind-config.yaml
docker build -t dlmac-task-runner:latest task-runner
kind load docker-image dlmac-task-runner:latest --name dlmac
```

**Datasets (`data/`)** — one row per stage:
`chain_id, arrival_time, edge_server_id, func_type, chain_deadline, profit, func_delay, mb, cores`.

- `chains_train.csv` / `chains_test.csv` — real Azure-derived chains (train / held-out).
- `staircase_chain.csv` (250×42) / `random_chain.csv` (250×40) — the shipped workflows in the same schema; regenerate with `python data/make_chain_datasets.py`.
- `graph.txt` — 30-node edge topology + gateways.

---

## 2. `lmac_dl.py` — DL-MAC trainer (real dataset)

Builds the environment from the chain CSVs, runs the baselines, trains the
DL-MAC multi-actor / centralized-critic policy (500 episodes, stabilized PPO:
`lr=1e-4, clip=0.15, ent=0.005, epochs=2`), and emits the paper's figures.

```bash
pip install -r requirements.txt
python lmac_dl.py           # or run on Colab: Runtime -> GPU, then run the cell
```

Point its `DATA` / `GRAPH` paths at `data/chains_train.csv` and `data/graph.txt`.

---

## 3. `run_dag_workflow.py` — experiment driver

Runs a workflow through the edge-server pool under a chosen policy, with measured
cold starts / compute / transport, and writes `results/*_summary.csv` and
`results/*_tasks.csv`.

```bash
# one workflow, one policy (local backend, no Docker needed)
python run_dag_workflow.py --workflow staircase_chain --policy dl_lru --instances 6

# every policy on a workflow
./run_all.sh staircase_chain 6
./run_all.sh random_chain 6
```

### Scheduling policies (`scheduler/policy.py`)

| Key | Method | Behaviour |
| --- | --- | --- |
| `s_cache` | S-Cache | run all stages on the home server (no cooperation) |
| `cocache` | CoCache | `K`-hop neighbour reuse of warm functions |
| `dsp` | DSP | demand-aware private/public placement |
| `pcache` | pCache | probabilistic placement (avoids hotspots) |
| `dl_lru` | DL-LRU | greedy dual-Lagrangian rule `argmin Q_i·w − V·π` |
| `dl_mac` | **DL-MAC (proposed)** | learned actor if supplied via `ctx["model"]`, else falls back to DL-LRU |
| `offline_opt` | Offline-Opt | least-congested feasible node (oracle proxy) |
