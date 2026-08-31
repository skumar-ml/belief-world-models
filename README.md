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
    <sup>1</sup>Univeristy og Illinois Urbana=Champaign &nbsp;&nbsp;&nbsp; <sup>2</sup>IBM
    <br>
    </br>
        <!-- <a href="#">
        <img src='https://img.shields.io/badge/arXiv-BB--WM-green' alt='Paper PDF'>
        </a> -->
        <!-- <a href='#'>
        <img src='https://img.shields.io/badge/Project_Page-BB--WM-blue' alt='Project Page'></a> -->
        <!-- <a href='#'>
        <img src='https://img.shields.io/badge/YouTube-BB--WM-rgb(255, 0, 0)' alt='Youtube'></a> -->
     </br>
    <!-- <table align="center">
        <img src="./figures/teaser.gif">
    </table> -->
</p>

Code for reproducing the main experiments in *Towards a Belief-Based World Model for LLM Agents* (BB-WM).

This repo evaluates **ReAct** and **ReflAct** agents on **ALFWorld** and **ScienceWorld** with four world-model conditions:

| Condition | ALFWorld | ScienceWorld |
|-----------|----------|--------------|
| Base agent | no WM | no WM |
| Belief | query-only belief WM | query-only belief WM |
| WALL-E | rule-based action validity | oracle action validity |
| BB-WM | Belief + WALL-E | Belief + WALL-E Oracle |

Built on [MPO](https://github.com/WeiminXiong/MPO) for ICL exemplars and dataset splits.

## Setup

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

## Running experiments

Every local run serves a policy model via **vLLM** and calls `eval_baselines.py`. API models (Sonnet) use a LiteLLM gateway instead.

### Single cell (smoke test)

```bash
# ALFWorld ReflAct + Belief WM smoke test (vLLM must already be running on :8000)
python eval_baselines.py \
  --exp_config alfworld_reflact_wm \
  --agent_config react_llama8b \
  --split test --max_steps 30 --debug --debug_n 3 \
  --api_base http://localhost:8000/v1 --api_key EMPTY
```

With vLLM already running:

```bash
METHOD=reflact_wm EXP_CONFIG=alfworld_reflact_wm \
  bash scripts/run_alfworld.sh

METHOD=sciworld_reflact_wm EXP_CONFIG=sciworld_reflact_wm \
  bash scripts/run_sciworld.sh
```

For Qwen3-14B:

```bash
MODEL=Qwen/Qwen3-14B AGENT_CONFIG=qwen3_14b \
  VLLM_ARGS="--gpu-memory-utilization 0.92 --max-model-len 8192 --enforce-eager" \
  METHOD=reflact_wm EXP_CONFIG=alfworld_reflact_wm \
  bash scripts/run_alfworld.sh
```

### Paper grids (3 seeds)

Reproduces Tables 1–3 from the paper (Llama + Qwen, 3 runs; Sonnet single run):

```bash
# Table 1 + Table 3 (ALFWorld): 60 cells total
bash scripts/run_paper_alfworld.sh

# Table 1 only (skip Memory ablation)
SKIP_MEMORY=1 bash scripts/run_paper_alfworld.sh

# Table 2 (ScienceWorld): 48 cells
bash scripts/run_paper_sciworld.sh

# Sonnet 4.6 via LiteLLM (set LITELLM_API_BASE / LITELLM_API_KEY)
bash scripts/run_paper_sonnet.sh
```

Override seeds or filter models:

```bash
RUNS="1 2 3" bash scripts/run_paper_alfworld.sh
MODELS_FILTER=qwen14b bash scripts/run_paper_sciworld.sh
```

### Sonnet (API)

Configure LiteLLM credentials, then:

```bash
export LITELLM_API_BASE=https://your-gateway/v1
export LITELLM_API_KEY=your-key

METHOD=reflact_wm EXP_CONFIG=alfworld_reflact_wm \
  bash scripts/run_api.sh
```

Or create `~/.litellm_env` with those variables.

## Output layout

```
outputs/
  alfworld/<agent>/<permutation>/<model>/unseen[/runN]/
  sciworld/<agent>/<permutation>/<model>/unseen[/runN]/
```

Each directory contains:
- `<task_id>.json` — per-episode trajectory
- `metrics.json` — aggregated SR@1, steps-to-goal (ALFWorld) or avg reward / AR-per-step (ScienceWorld)
- `run_config.json` — provenance (configs + CLI args)
- `log.txt` — run log

**Pre-computed trajectories** from the paper are not included in this repo; they will be hosted separately (Drive/Dropbox).

## Method × config reference

| Paper method | `--exp_config` (ALFWorld) | `--exp_config` (ScienceWorld) |
|--------------|---------------------------|-------------------------------|
| ReAct | `alfworld` | `sciworld_react` |
| ReAct + Belief | `alfworld_react_wm` | `sciworld_react_wm` |
| ReAct + WALL-E | `alfworld_react_walle` | `sciworld_react_walle_oracle` |
| ReAct + BB-WM | `alfworld_react_walle_wm` | `sciworld_react_walle_oracle_wm` |
| ReflAct | `alfworld_reflact` | `sciworld_reflact` |
| ReflAct + Belief | `alfworld_reflact_wm` | `sciworld_reflact_wm` |
| ReflAct + WALL-E | `alfworld_reflact_walle` | `sciworld_reflact_walle_oracle` |
| ReflAct + BB-WM | `alfworld_reflact_walle_wm` | `sciworld_reflact_walle_oracle_wm` |
| Memory ablation (Table 3) | `alfworld_react_wm_det`, `alfworld_reflact_wm_det` | — |

Models: `react_llama8b` (Llama-3.1-8B-Instruct), `qwen3_14b` (Qwen3-14B), `litellm_claude_sonnet` (Sonnet 4.6).

## Project layout

| Directory | Contents |
|-----------|----------|
| `agents/` | LLM agent loop (OpenAI SDK → vLLM or LiteLLM) |
| `wm/` | Belief-state world model + WALL-E rules |
| `envs/` | Environment wrappers (ALFWorld, SciWorld, WM, WALL-E) |
| `configs/` | Task and model configs |
| `prompt/` | Instruction templates + per-task ICL exemplars |
| `tasks/` | Task loaders |
| `scripts/` | Data download + experiment runners |
| `data/sciworld/` | Split indices and per-task step budgets |

<!-- ## License

Add your chosen license file before making this repository public. -->
