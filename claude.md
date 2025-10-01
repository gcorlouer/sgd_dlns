# CLAUDE.md — DLN Experiments (Agent Guide)

You are **Claude Code** assisting on a Deep Linear Networks (DLN) research repo. Follow these rules strictly.

---

## 🚦 Branch & Workflow Policy (IMPORTANT)

- **Default working branch:** `dev`
- **NEVER push to `main`**. I (the human) will merge/push `main` after review.
- Flow:
  1) Commit to `dev`
  2) Open a PR from `dev` → `main`
  3) Wait for human review; **do not merge** the PR yourself
- Temporary branches are fine; base them on `dev` and merge back to `dev`.

**Allowed git commands:** `git status`, `git add -p`, `git commit`, `git restore -p`, `git push origin dev`, `git pull --rebase`, `git log -p -- <file>`, `git diff`, `gh pr create --base main --head dev --fill`  
**Forbidden:** `git push origin main`, merging to `main`.

---

## 🧭 Repo Map

data/ # datasets or generated data
results/ # artifacts, figures (loss curves, etc.)
scripts/
models.py # model definitions (DLN, helpers)
teacher.py # teacher generation & dataset logic
train.py # training loop / experiment runner (argparse)
mode_alignment.py # (WIP) alignment analysis
scratch_pad.py # ad-hoc experiments; keep isolated
wandb/ # Weights & Biases runs/logs
environment.yml # conda environment spec


**Primary code paths:** `scripts/models.py`, `scripts/teacher.py`, `scripts/train.py`.

---

## 🛠 Environment & Tooling

**Conda / Python**
bash
# first time
conda env create -f environment.yml
conda activate dln   # or the name specified in environment.yml

# update on changes
conda env update -f environment.yml --prune


W&B

Default to offline for quick iteration:

export WANDB_MODE=offline


To log online: wandb login (never hard-code keys).

Formatting/linting
black scripts
ruff check scripts


▶️ Common Commands

Discover training args

python -m scripts.train --help


Tiny smoke run (adjust to your CLI)

python -m scripts.train \
  --iters 1000 \
  --gamma 1.5 \
  --lr 1e-4 \
  --batch-size 1


Save figures

Write plots to results/ with descriptive names:

results/<exp>_iter_<N>_gamma_<g>_lr_<lr>_batch_<b>_loss.png

📓 CLAUDE.md Usage & Tuning

This file is automatically pulled into context. Keep it concise and high-signal.

Add recurring commands, pitfalls, and conventions as they stabilize.

You may update this file during work; include changes in commits.

🔐 Permissions & Safety (Allowlist)

Always-allow suggestions (edit to match your risk tolerance):

Edit

Bash(python*|pytest*|conda*|ruff*|black*|wandb*)

Bash(git add:*|git commit:*|git push origin dev)

Bash(gh pr create*)

Ask before:

Modifying environment.yml

Installing new global packages

Deleting files / moving directories

Running long jobs or networked tools beyond the codebase

Never:

Push or merge to main

Exfiltrate secrets or tokens

🧪 Coding Conventions (Python)

Prefer functional cores with thin CLI wrappers.

Add --seed to training and record seeds/configs for reproducibility.

Type hints where practical; docstrings for public APIs.

Explicit shapes in comments for tensors.

Small, reviewable diffs; prefer adding functions over rewriting modules.

📈 DLN Observables & Research Hygiene

When implementing or plotting, consider tracking:

Loss curves per config (already in results/)

Mode growth along teacher singular vectors

Alignment (angles/cosine with teacher modes)

Effective noise vs drift estimates (if/when implemented)

Add reusable metrics in scripts/metrics.py (create if missing) rather than embedding logic in train.py.

✅ Proven Workflows
A) Explore → Plan → Code → Commit

Explore: read relevant files; gather constraints. Do not write code yet.

Plan: propose a short plan (bullets). Use the keyword “think” before coding.

Implement: minimal surface area; prefer targeted changes.

Verify: run the smallest adequate check (pytest subset, short training).

Commit to dev; optionally open a PR dev → main.

B) TDD Loop (tests first)

Write tests (no implementation changes yet).

Run and confirm they fail.

Commit tests.

Implement code until tests pass.

Commit to dev, open PR if substantial.

C) Visual / Artifacts Loop

If reproducing figures or UI-like outputs, iterate with saved images in results/, comparing against a target.

D) Safe-YOLO (rare; use sandbox)

claude --dangerously-skip-permissions only inside an isolated container with no internet access, for bulk lint fixes or boilerplate generation. Otherwise ask first.

🔍 Codebase Q&A

You may search/explain internals agentically:

“How does logging/training loop work?”

“Where is teacher rank parameterized?”

“What edge cases does X handle?”

Prefer reading code and git history over guessing.

🧰 Git & GitHub

Create clear commits and PRs with diffs easy to review.

Use gh pr create --base main --head dev --fill for PRs.

For reviews/fixes, apply changes to dev. Do not merge to main.

Commit message style

feat(train): add gamma sweep and plot saver

- introduce --gamma-list CLI flag
- save loss curves to results/iter_<N>_gamma_<g>_...
- refactor models.DLN.forward for clarity

Refs: #123


Use feat|fix|refactor|docs|test|chore(scope): summary.

🧑‍🏫 Prompting Conventions

Be specific: target files, functions, and acceptance criteria.

Mention exact files/folders to read or update.

Use images/plots when relevant (paste path or file).

Use /clear between unrelated tasks to keep context tight.

Course correct early: propose a plan; await confirmation before large edits.

For complex tasks, create a Markdown checklist and work it down.

🧵 Passing Data / Context

You may: read files, fetch URLs (if allowed), run --help, or pipe logs into context (e.g., cat logs.txt | claude).

Prefer small, relevant excerpts; avoid flooding context.

🧪 Jupyter

prefer .py file to run things

🧱 Multi-Claude & Parallelization (Optional)

Separate sessions for writer vs reviewer improves quality.

Use git worktrees for parallel tasks:

git worktree add ../repo-feature-a feature-a
cd ../repo-feature-a && claude


Keep one terminal/tab per worktree; clean up when done.

🧰 Headless Mode (Automation)

Non-interactive runs:

claude -p "<prompt>" --allowedTools Edit Bash(git commit:*) --output-format stream-json


Good for: issue triage, naming lint, scripted refactors. Headless sessions do not persist.

Hard Rules (do not violate)

Never push to main or merge PRs into main.

Ask before altering environment.yml or installing global deps.

Keep experiments reproducible (record configs & seeds).

Don’t touch files outside this repo.

Prefer smaller, reviewable diffs with clear commit messages.

