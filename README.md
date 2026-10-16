# Reproducing COCONUT and Testing Written Steps Between Latent Thoughts

Course project by Eran Ben Eliyahou (Technion). The report is `report/report.pdf` (LaTeX source next to it).

- **Part 1** reproduces COCONUT (Hao et al., 2024) on GSM8K at inference time with released GPT-2 models, plus a
  control without the hidden-state feedback, and reproduces SIM-CoT (Wei et al., 2026).
- **Part 2** tests a small version of "thinking in turns": the model thinks, writes down its first step, and thinks
  again with that step in its input. Experiment 1 gives the step to the released model; Experiment 2 trains the
  model briefly on its own written steps.

All numbers below are on the full GSM8K test set (1319 questions), greedy decoding, seed 0, one NVIDIA L40S GPU.

## Main results

**Part 1** (`coconut/results/gpt2/results.json`, `coconut/results/simcot/simcot_run.json`)

| Continuous thoughts | 0 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| Coconut GPT-2 (paper: 34.1% at 6) | 10.4 | 12.9 | 21.0 | 27.1 | | |
| Coconut GPT-2, no feedback (plain slots) | 10.4 | 10.2 | 9.9 | 9.6 | | |
| SIM-CoT GPT-2 (paper: 44.8% at 10) | 5.4 | 13.6 | 27.9 | 38.2 | 43.5 | 44.7 |

**Part 2** (SIM-CoT, 10 thoughts per round; `coconut/results/simcot/simcot_run.json`, `simcot_t2in.json`, `ft_eval.json`)

| Model | No step | Own step | True step | Own step inside | True step inside |
|---|---|---|---|---|---|
| Released model (Experiment 1) | 44.7 | 34.4 | 38.6 | 24.5 | 27.2 |
| FT-plain: short training without steps (Experiment 2) | 45.5 | 35.5 | 39.1 | | |
| FT-mix: short training with its own steps (Experiment 2) | 45.3 | 41.5 | 45.4 | | |

## What is in this folder

```
README.md                  this file
environment.yml            Python packages and versions (conda env "neuroAI_project")
report/                    report.pdf + LaTeX source (main.tex, references.bib, cogsci.sty)
coconut/
  coconut.py dataset.py utils.py run.py LICENSE README.md
                           official Coconut code (github.com/facebookresearch/coconut), unchanged
  args/gsm_coconut.yaml    official GSM8K config (read by run_native.py)
  preprocessing/gsm_icot.* official script that builds the GSM8K data files
  data/gsm8k_test.json     GSM8K test set in HF format (used by run_fewshot.py)
  eval_qwen_1.5b_gsm8k_*.py   two scripts from our first attempt, kept only because smoke_test.py (check C0)
                           compares our few-shot prompt with theirs byte for byte
  experiments/             our code (below)
  results/                 every result file used in the report, with per-question records
logs/                      the Slurm logs of every run in the report
```

### Our code (`coconut/experiments/`)

| File | What it does |
|---|---|
| `common.py` | loading the models (Coconut GPT-2, SIM-CoT + explainer, Qwen exports), filling thought slots, greedy decoding |
| `score.py` | accuracy, Wilson intervals, exact McNemar test, paired bootstrap |
| `run_native.py` | Part 1: 0 to 6 thoughts in the official format; `--fill plain` = control without feedback |
| `smoke_test.py` | checks before the GPT-2 run (our code = official code, prompt = first attempt's prompt, ...) |
| `prompts.py`, `run_fewshot.py` | the few-shot two-stage prompts (attempt 7 in the report, floor effect) |
| `qwen_check.py`, `qwen3_check.py` | 10-minute checks of two public Qwen Coconut models (attempts 8 and 9) |
| `simcot_run.py` | SIM-CoT: checks, 0 to 10 thoughts (Part 1), no step / own step / true step (Experiment 1) |
| `simcot_t2in.py` | Experiment 1: the step written inside the question |
| `simcot_ft_steps.py` | Experiment 2, step 0: the model's own first step for 50,000 training questions |
| `simcot_finetune.py` | Experiment 2: trains FT-mix (`--mode mix`) or FT-plain (`--mode plain`) |
| `simcot_ft_eval.py` | Experiment 2: tests the original, FT-plain and FT-mix models |
| `*.slurm` | the Slurm jobs that ran everything (below) |

Code names in files vs. names in the report: `L` = no step, `T2` = own step, `T2_gold` = true step,
`T2_in` = step inside the question, `FT_mix` / `FT_plain` = the two trained models.

## Which job made which result

| Report item | Slurm job | Result files | Our run (log id) | Time |
|---|---|---|---|---|
| Fig. 1 COCONUT curve; Table 3 row 7 (few-shot) | `run_gpt2.slurm` | `results/gpt2/native.json`, `fewshot_S*.json`, `results.json` | 1449565 | ~10 min |
| Fig. 1 no-feedback control; Table 3 row 8 | `qwen_check.slurm` (needs `run_gpt2` first) | `results/gpt2/native_plain.json`, `results.json`; `results/qwen7b/check/` | 1449567 | < 1 h |
| Table 3 row 9 | `qwen3_check.slurm` | `results/qwen3/check/` | 1449576 | < 1 h |
| Fig. 1 SIM-CoT curve; Experiment 1 (step after the question): Fig. 2, Table 1 | `simcot_run.slurm` | `results/simcot/simcot_run.json` | 1449771 | ~15 min |
| Experiment 1 (step inside the question): Fig. 2, Table 1 | `simcot_t2in.slurm` (needs `simcot_run` first) | `results/simcot/simcot_t2in.json` | 1450454 | ~7 min |
| Experiment 2: Fig. 3, Table 2 | `simcot_ft.slurm` (needs `simcot_run` first) | `results/simcot/ft_train_steps.json`, `ft_train_mix.json`, `ft_train_plain.json`, `ft_eval.json` | 1450503 | ~6.5 h |

Each job writes `logs/out_job<id>.txt` (progress) and one log per step in `logs/` (details, checks).

## How to reproduce

Run everything from the folder that contains `coconut/` and `logs/` (this folder).

1. **Environment** (Python 3.12):
   ```bash
   conda create -y -n neuroAI_project -c conda-forge python=3.12 pip
   conda activate neuroAI_project
   pip install torch==2.6.0 numpy==2.1.3 transformers==4.52.4 datasets==3.6.0 tokenizers==0.21.1 tqdm==4.67.0 pyyaml scipy
   ```
2. **Data** (GSM8K in the Coconut format: `coconut/data/gsm_{train,valid,test}.json`):
   ```bash
   cd coconut && bash preprocessing/gsm_icot.bash && cd ..
   ```
3. **Models** (once, on a machine with internet; the jobs run offline from the Hugging Face cache):
   ```bash
   cd coconut
   python experiments/common.py      # GPT-2 + tsrigo/coconut stage4_checkpoint_25, ends with "loaded OK"
   python -c "from huggingface_hub import hf_hub_download as d; d('internlm/SIM_COT-GPT2-Coconut','checkpoint_28'); d('internlm/SIM_COT-GPT2-Coconut','config.json')"
   # only for attempts 8 and 9:
   python -c "from huggingface_hub import snapshot_download as s; s('agurung/coconut-qwen2.5-7b'); s('agurung/coconut-qwen3-4b-ff'); s('Qwen/Qwen3-4B-Instruct-2507', allow_patterns=['*.json','*.txt'])"
   cd ..
   ```
4. **Run the jobs** (one GPU with at least 24 GB; we used an L40S 48 GB):
   ```bash
   mkdir -p logs
   sbatch coconut/experiments/run_gpt2.slurm
   sbatch coconut/experiments/qwen_check.slurm      # after run_gpt2 has finished
   sbatch coconut/experiments/qwen3_check.slurm
   sbatch coconut/experiments/simcot_run.slurm
   sbatch coconut/experiments/simcot_t2in.slurm     # after simcot_run has finished
   sbatch coconut/experiments/simcot_ft.slurm       # after simcot_run has finished
   ```
   Before submitting, edit the Slurm headers for your cluster: the GPU name (`--gres=gpu:L40:1`), the mail address,
   and the conda path (`/home/eran.b/miniconda3`). Without Slurm, a job can be run directly, e.g.
   `SLURM_SUBMIT_DIR=$PWD SLURM_JOB_ID=local bash coconut/experiments/simcot_run.slurm`.

## Notes

- Every job starts with checks and stops if one fails, for example: our thought filling gives the same vectors as
  the official code, our decoding gives the same tokens as the official `generate`, training inputs are identical to
  test inputs, and re-running the original SIM-CoT model gives exactly the earlier counts (589 / 454 / 509 correct).
- Inference is greedy and was exactly repeatable across our jobs. Training on a GPU may differ slightly between runs.
- The two trained models (~0.5 GB each, `coconut/checkpoints_ft/`) are not included; `simcot_ft.slurm` recreates them.
- Attempts 1 to 6 in the report (Table 3) used older code that is not included here; this repository holds only the
  final version.
- Use of AI: see the Acknowledgments section of the report.
