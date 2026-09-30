<p align="center">

  <h2 align="center">Towards a Belief-Based World Model <br>for LLM Agents</h2>
  <p align="center">
    <a href="https://skumar-ml.github.io/"><strong>Shubham Kumar</strong></a><sup>1</sup>
    ·
    <a href="https://research.ibm.com/people/harshit-kumar"><strong>Harshit Kumar</strong></a><sup>2</sup>
    ·
    <a href="https://vision.ai.illinois.edu/narendra-ahuja/"><strong>Narendra Ahuja</strong></a><sup>1</sup>
    ·
    <a href="https://saurabhjha.one/"><strong>Saurabh Jha</strong></a><sup>2</sup>
    <br>
    <sup>1</sup>University of Illinois Urbana-Champaign &nbsp;&nbsp;&nbsp; <sup>2</sup>IBM
    <br>
    </br>
    <a href="https://arxiv.org/abs/2609.00455">
      <img src="https://img.shields.io/badge/arXiv-BB--WM-green" alt="Paper PDF">
    </a>
        <!-- <a href='#'>
        <img src='https://img.shields.io/badge/Project_Page-BB--WM-blue' alt='Project Page'></a> -->
     </br>
</p>

Code for reproducing *Towards a Belief-Based World Model for LLM Agents* (BB-WM).

This repo evaluates **ReAct** and **ReflAct** agents on **ALFWorld**, **ScienceWorld**, and **BabyAI-Text** under four world-model conditions:

| Condition | ALFWorld | ScienceWorld | BabyAI-Text |
|-----------|----------|--------------|-------------|
| Original | no WM | no WM | no WM |
| Belief | query-only belief WM | query-only belief WM | query-only 6×6 placement-prior WM |
| WALL-E | rule-based action validity | oracle action validity | oracle action validity |
| BB-WM | Belief + WALL-E | Belief + WALL-E Oracle | Belief + WALL-E Oracle |

Agents are frozen (temperature 0). Variants differ only in the world-model interface. Built on [MPO](https://github.com/WeiminXiong/MPO) for ICL exemplars and ALFWorld / ScienceWorld splits. BabyAI-Text follows the [BALROG](https://github.com/balrog-ai/BALROG) MixedTrainLocal protocol with frozen `(family, seed)` indices.

Paper figures plot **reward-within-budget** (success for ALFWorld / BabyAI-Text; anytime return for ScienceWorld). Llama-3.1-8B-Instruct and Qwen3-14B use 3 seeds; Sonnet 4.6 uses 1 seed. Qwen3-14B is run with thinking off.

## Setup

### Local GPU / vLLM

```bash
conda create -n bbwm python=3.10 -y
conda activate bbwm
pip install -r requirements.txt
bash scripts/download_data.sh   # ALFWorld game files + env libs (~2 GB)
```

**Requirements:**
- NVIDIA GPU with enough VRAM for vLLM (40 GB recommended for Qwen3-14B)
- Java 17+ for ScienceWorld (`openjdk-17`)
- Hugging Face token for gated models (Llama-3.1-8B-Instruct)

Set `HF_TOKEN` or place a token at `~/.cache/huggingface/token`.

BabyAI-Text needs no extra download: frozen indices are in `data/babyai/`, and MiniGrid comes from `requirements.txt` ([BartekCupial fork](https://github.com/BartekCupial/Minigrid)). ScienceWorld ships with the `scienceworld` package.

### CPU / OpenRouter (no local GPU)

```bash
bash scripts/setup_openrouter.sh   # Miniforge env + Java 17 + API deps + ALFWorld data
source scripts/env_openrouter.sh   # conda activate bbwm + JAVA_HOME
# OPENROUTER_API_KEY must be set (e.g. via ~/.config/secrets.env)
```

`scripts/env_openrouter.sh` sets `SDL_VIDEODRIVER=dummy` so MiniGrid can run headless.

## Running experiments

Every local GPU run serves a policy via **vLLM** and calls `eval_baselines.py`. OpenRouter and Sonnet skip vLLM and call the same harness.

Prefix any `scripts/run_paper_*.sh` command with `BACKEND=openrouter` to use OpenRouter instead of vLLM.

### Smoke tests

```bash
# GPU / vLLM already running on :8000
python eval_baselines.py \
  --exp_config alfworld_reflact_wm \
  --agent_config react_llama8b \
  --split test --max_steps 30 --debug --debug_n 3 \
  --api_base http://localhost:8000/v1 --api_key EMPTY

METHOD=reflact_wm EXP_CONFIG=alfworld_reflact_wm \
  bash scripts/run_alfworld.sh

METHOD=sciworld_reflact_wm EXP_CONFIG=sciworld_reflact_wm \
  bash scripts/run_sciworld.sh

METHOD=react EXP_CONFIG=babyai_react \
  bash scripts/run_babyai.sh
```

OpenRouter:

```bash
METHOD=reflact_wm EXP_CONFIG=alfworld_reflact_wm AGENT_CONFIG=openrouter_llama8b \
  DEBUG=1 bash scripts/run_openrouter.sh

METHOD=sciworld_reflact_wm EXP_CONFIG=sciworld_reflact_wm AGENT_CONFIG=openrouter_qwen3_14b \
  DEBUG=1 bash scripts/run_openrouter.sh

METHOD=react EXP_CONFIG=babyai_react AGENT_CONFIG=openrouter_llama8b \
  DEBUG=1 bash scripts/run_openrouter.sh
```

Qwen3-14B on GPU:

```bash
MODEL=Qwen/Qwen3-14B AGENT_CONFIG=qwen3_14b \
  VLLM_ARGS="--gpu-memory-utilization 0.92 --max-model-len 8192 --enforce-eager" \
  METHOD=reflact_wm EXP_CONFIG=alfworld_reflact_wm \
  bash scripts/run_alfworld.sh
```

### Figure 3 — main grid (ALFWorld, ScienceWorld, BabyAI-Text)

Llama + Qwen, 3 seeds, 2 agents × 4 world-model conditions.

```bash
# ALFWorld: 134 unseen games, 30 env steps (also runs Figure 4 Memory cells)
bash scripts/run_paper_alfworld.sh

# ALFWorld Figure 3 only
SKIP_MEMORY=1 bash scripts/run_paper_alfworld.sh

# ScienceWorld: 211 unseen episodes, per-task budgets in data/sciworld/max_steps.json
bash scripts/run_paper_sciworld.sh

# BabyAI-Text: 100 tasks (goto, pickup, putnext, pick_up_seq_go_to; omits open), 64 env steps
bash scripts/run_paper_babyai.sh
```

BabyAI-Text is zero-shot (no ICL) on `BabyAI-MixedTrainLocal-v0` with `num_dists=0`. The full five-family index is `data/babyai/test_indices.json`; paper runs pass `--families goto+pickup+putnext+pick_up_seq_go_to`. Belief / BB-WM use a uniform prior over the 6×6 walkable interior once the room is localized.

Override seeds or models:

```bash
RUNS="1 2 3" bash scripts/run_paper_alfworld.sh
MODELS_FILTER=qwen14b bash scripts/run_paper_sciworld.sh
MODELS_FILTER=llama8b bash scripts/run_paper_babyai.sh
```

### Figure 4 — Memory vs. Belief

Memory (`wm_det`) keeps Belief’s deterministic state but drops the distribution over unobserved locations. ALFWorld Memory cells are included in `run_paper_alfworld.sh` by default. ScienceWorld:

```bash
SKIP_MEMORY=0 bash scripts/run_paper_sciworld.sh
```

### Figure 5 — Zipf (non-uniform) ALFWorld priors

Target objects are placed with Zipf(\(α=1.25\)) over valid receptacles (`wm/zipf_ranks.json`). The first run builds `data/alfworld_zipf/` and writes evals under `outputs/alfworld/.../unseen_zipf/`. Figure 5 compares Original, Belief, and Belief-NoProb:

```bash
SKIP_MEMORY=1 INCLUDE_NO_PROB=1 PLACEMENT=zipf \
  bash scripts/run_paper_alfworld.sh
```

This also evaluates WALL-E / BB-WM on the Zipf games; the figure only uses `original`, `wm`, and `wm_no_prob`. Overlay the uniform Original curve when plotting (see below).

### Sonnet 4.6 (Figure 3, one seed)

```bash
export LITELLM_API_BASE=https://your-gateway/v1
export LITELLM_API_KEY=your-key
# or: create ~/.litellm_env with those variables

bash scripts/run_paper_sonnet.sh              # ALFWorld + ScienceWorld + BabyAI
BENCHMARK=babyai bash scripts/run_paper_sonnet.sh
```

Single cell:

```bash
METHOD=reflact_wm EXP_CONFIG=alfworld_reflact_wm bash scripts/run_api.sh
METHOD=react EXP_CONFIG=babyai_react bash scripts/run_api.sh
```

### Figures from dumps

```bash
pip install matplotlib
python scripts/plot_success_within_budget.py --benchmarks alfworld,sciworld,babyai
```

Writes `figures/success_within_budget/`:
- `combined_sr_within_normalized_budget.{pdf,png}` — Figure 3 layout
- `<bench>_sr_within_budget.{pdf,png}` plus ScienceWorld anytime-return vs. budget fraction
- `<bench>_summary.csv` and `<bench>_curves.json`

```bash
# Figure 4
python scripts/plot_success_within_budget.py --benchmarks alfworld,sciworld \
  --methods original,wm,wm_det --models Llama-3.1-8B-Instruct,Qwen3-14B

# Figure 5 (Zipf vs. uniform Original)
python scripts/plot_success_within_budget.py --benchmarks alfworld \
  --split unseen_zipf --methods original,wm,wm_no_prob \
  --reference-split unseen --reference-methods original
```

## Output layout

```
outputs/
  alfworld/<agent>/<permutation>/<model>/unseen[/runN]/
  alfworld/<agent>/<permutation>/<model>/unseen_zipf[/runN]/   # Figure 5
  sciworld/<agent>/<permutation>/<model>/unseen[/runN]/
  babyai/<agent>/<permutation>/<model>/unseen[/runN]/
```

Each directory contains `<task_id>.json` trajectories, `metrics.json`, `run_config.json`, and `log.txt`.

Permutations: `original` (ALFWorld / BabyAI) or `regular` (ScienceWorld), `wm` (Belief), `walle` / `walle_oracle` (WALL-E), `wm_walle` / `walle_oracle_wm` (BB-WM), `wm_det` (Memory), `wm_no_prob` (Belief-NoProb).

**Pre-computed trajectories** from the paper are not included in this repo; they will be hosted separately (Drive/Dropbox).

## Method × config reference

| Paper method | `--exp_config` (ALFWorld) | `--exp_config` (ScienceWorld) | `--exp_config` (BabyAI-Text) |
|--------------|---------------------------|-------------------------------|------------------------------|
| ReAct | `alfworld` | `sciworld_react` | `babyai_react` |
| ReAct + Belief | `alfworld_react_wm` | `sciworld_react_wm` | `babyai_react_wm` |
| ReAct + WALL-E | `alfworld_react_walle` | `sciworld_react_walle_oracle` | `babyai_react_walle_oracle` |
| ReAct + BB-WM | `alfworld_react_walle_wm` | `sciworld_react_walle_oracle_wm` | `babyai_react_walle_oracle_wm` |
| ReflAct | `alfworld_reflact` | `sciworld_reflact` | `babyai_reflact` |
| ReflAct + Belief | `alfworld_reflact_wm` | `sciworld_reflact_wm` | `babyai_reflact_wm` |
| ReflAct + WALL-E | `alfworld_reflact_walle` | `sciworld_reflact_walle_oracle` | `babyai_reflact_walle_oracle` |
| ReflAct + BB-WM | `alfworld_reflact_walle_wm` | `sciworld_reflact_walle_oracle_wm` | `babyai_reflact_walle_oracle_wm` |
| Memory | `alfworld_react_wm_det`, `alfworld_reflact_wm_det` | `sciworld_react_wm_det`, `sciworld_reflact_wm_det` | — |
| Belief-NoProb | `alfworld_react_wm_no_prob`, `alfworld_reflact_wm_no_prob` | — | — |

`--method` is a metrics label only; the output path comes from `benchmark` / `agent` / `permutation` in the task config.

Models: `react_llama8b` / `openrouter_llama8b` (Llama-3.1-8B-Instruct), `qwen3_14b` / `openrouter_qwen3_14b` (Qwen3-14B, thinking off), `litellm_claude_sonnet` (Sonnet 4.6).

ALFWorld WALL-E uses the released neurosymbolic rules (`wm/walle/SOURCE.md`). ScienceWorld and BabyAI-Text use an oracle validity check against the simulator. Episodes stop early after 10 consecutive invalid actions or world-model queries.

## Project layout

| Directory | Contents |
|-----------|----------|
| `agents/` | LLM agent loop (OpenAI SDK → vLLM, OpenRouter, or LiteLLM) |
| `wm/` | Belief-state world model + WALL-E rules + BabyAI 6×6 prior |
| `envs/` | Environment wrappers (ALFWorld, SciWorld, BabyAI, WM, WALL-E) |
| `configs/` | Task and model configs |
| `prompt/` | Instruction templates + per-task ICL exemplars |
| `tasks/` | Task loaders (including Zipf ALFWorld cache) |
| `scripts/` | Data download, paper grids, figure script |
| `data/sciworld/` | Split indices and per-task step budgets |
| `data/babyai/` | Frozen BALROG `(family, seed)` indices |

<!-- ## License

Add your chosen license file before making this repository public. -->
