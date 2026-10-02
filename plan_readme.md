# 📌 DL Project — Team Collaboration Plan

> **Repo:** [https://github.com/Anshul-0211/DL_project](https://github.com/Anshul-0211/DL_project)  
> **Repo Owner:** Anshul  
> **Collaborators:** Ayush · Naman · Kushal  

---

## 🗺️ Overview

This document defines the Git workflow and tasks for every team member.  
**Read only your section.** You can hand your section directly to Claude and it will execute the steps for you.

### Branch Structure
| Branch | Owner |
|--------|-------|
| `main` | Protected — merged into via Pull Requests |
| `Anshul` | Anshul's working branch |
| `Ayush` | Ayush's working branch |
| `Naman` | Naman's working branch |
| `Kushal` | Kushal's working branch |

### Workflow at a Glance
```
Ayush pushes full project to main (3–4 commits)
         ↓
All teammates clone repo & switch to their branch
         ↓
Each person makes small commits on their branch
         ↓
Each person opens a Pull Request → main
         ↓
Anshul reviews & merges all PRs
```

---

---

## 👤 AYUSH — Code Owner & Initial Committer

> **You have the full project codebase. Your job is to push it to `main` in stages (3–4 commits) to build a meaningful commit history. You also need to build your own commit history on the `Ayush` branch.**

### ✅ Prerequisites
- Accept the GitHub collaborator invite (check your email / GitHub notifications)
- Make sure `git` is installed on your system

### Step 1 — Clone the repository
```bash
git clone https://github.com/Anshul-0211/DL_project.git
cd DL_project
```

### Step 2 — Push the full project to `main` in 3–4 staged commits

> 💡 **If you're using Claude:** Tell Claude — *"I have a complete DL project codebase. Help me push it to the `main` branch of this repo in 3–4 logical, staged commits that build a realistic commit history. Look at my files and decide the best groupings (e.g. data → model → training → evaluation). The repo is already cloned at [path]."*

Do NOT push everything in one commit. Stage your files in logical groups:

**Suggested grouping (Claude/you will adapt based on actual files):**
```bash
# Commit 1 — Project setup & data
git add data/ requirements.txt .gitignore   # adjust filenames as needed
git commit -m "feat: add dataset and preprocessing pipeline"

# Commit 2 — Model architecture
git add models/
git commit -m "feat: add model architecture"

# Commit 3 — Training scripts
git add train.py config.py   # adjust filenames as needed
git commit -m "feat: add training loop and configuration"

# Commit 4 — Evaluation & results
git add evaluate.py results/ notebooks/   # adjust filenames as needed
git commit -m "feat: add evaluation scripts and results"

# Push to main
git push origin main
```

### Step 3 — Build your own branch history on `Ayush`

After pushing to `main`, switch to your personal branch and add your own contributions:

```bash
git checkout Ayush
git merge main   # bring main's code into your branch
```

> 💡 **If you're using Claude:** Tell Claude — *"Now that the main code is pushed, help me make 2–3 small but meaningful commits on my `Ayush` branch. Look at the codebase and suggest improvements — maybe tune a hyperparameter, add a utility function, fix an issue, or add a notebook. Commit each change separately with a good commit message."*

Once done:
```bash
git push origin Ayush
```

### Step 4 — Open a Pull Request
1. Go to [https://github.com/Anshul-0211/DL_project](https://github.com/Anshul-0211/DL_project)
2. Click **"Compare & pull request"** next to your `Ayush` branch
3. Title: `[Ayush] <brief description of your contributions>`
4. In the description, write a short summary of what you changed/added on your branch
5. Click **"Create pull request"**

---

---

## 👤 ANSHUL — Repo Owner & Reviewer

> **You own the repo. Your job is to work on your `Anshul` branch (small contributions), open a Pull Request, AND review + merge everyone else's PRs.**

### ✅ Prerequisites
- You've already set up the local repo (done ✅)
- Wait for **Ayush to complete Step 2** (push the full project to `main`) before starting your contributions

### Step 1 — Pull the latest main and update your branch
```bash
git checkout main
git pull origin main
git checkout Anshul
git merge main
```

### Step 2 — Make 2–3 small commits on your `Anshul` branch

> 💡 **If you're using Claude:** Tell Claude — *"Look at the DL project codebase on the `Anshul` branch. Help me make 2–3 small, meaningful commits — for example: add comments/docstrings, fix a bug, improve logging, add a visualization, or clean up code. Make each change a separate commit with a descriptive message."*

```bash
# After each change:
git add <changed files>
git commit -m "your descriptive message"
```

### Step 3 — Push your branch
```bash
git push origin Anshul
```

### Step 4 — Open a Pull Request
1. Go to [https://github.com/Anshul-0211/DL_project](https://github.com/Anshul-0211/DL_project)
2. Click **"Compare & pull request"** next to your `Anshul` branch
3. Title: `[Anshul] <brief description of your contributions>`
4. Write a short description of your changes
5. Click **"Create pull request"**

### Step 5 — Review & Merge everyone's Pull Requests *(Repo Owner task)*

Once all teammates have opened their PRs:
1. Go to the **"Pull requests"** tab on the repo
2. Open each PR one by one
3. Review the changes (click **"Files changed"**)
4. If everything looks good, click **"Merge pull request"** → **"Confirm merge"**
5. Repeat for all open PRs

> ⚠️ Merge in this order to avoid conflicts: `Ayush` → `Naman` → `Kushal` → `Anshul`

---

---

## 👤 NAMAN — Contributor

> **Your job is to work on the `Naman` branch, make 2–3 small meaningful commits, and open a Pull Request to `main`.**

### ✅ Prerequisites
- Accept the GitHub collaborator invite (check your email / GitHub notifications)
- Make sure `git` is installed

### Step 1 — Clone the repository
```bash
git clone https://github.com/Anshul-0211/DL_project.git
cd DL_project
```

> ⚠️ **Wait until Ayush has pushed the project to `main`** before proceeding. You'll know it's ready when you can see the project files at [https://github.com/Anshul-0211/DL_project](https://github.com/Anshul-0211/DL_project).

### Step 2 — Switch to your branch
```bash
git checkout Naman
git merge main   # bring the latest code into your branch
```

### Step 3 — Make 2–3 small commits

> 💡 **If you're using Claude:** Tell Claude — *"I'm working on the `Naman` branch of a Deep Learning project at this path: [your local path]. Look at the codebase and help me make 2–3 small, meaningful commits — for example: add comments, improve a function, add a plot, fix an edge case, or add a helper script. Make each change a separate git commit with a good message."*

```bash
# After each change:
git add <changed files>
git commit -m "your descriptive message"
```

### Step 4 — Push your branch
```bash
git push origin Naman
```

### Step 5 — Open a Pull Request
1. Go to [https://github.com/Anshul-0211/DL_project](https://github.com/Anshul-0211/DL_project)
2. You'll see a banner: **"Naman had recent pushes"** → Click **"Compare & pull request"**
3. Title: `[Naman] <brief description of your contributions>`
4. Write a short description of what you did
5. Set base branch to `main`
6. Click **"Create pull request"**

---

---

## 👤 KUSHAL — Contributor

> **Your job is to work on the `Kushal` branch, make 2–3 small meaningful commits, and open a Pull Request to `main`.**

### ✅ Prerequisites
- Accept the GitHub collaborator invite (check your email / GitHub notifications)
- Make sure `git` is installed

### Step 1 — Clone the repository
```bash
git clone https://github.com/Anshul-0211/DL_project.git
cd DL_project
```

> ⚠️ **Wait until Ayush has pushed the project to `main`** before proceeding.

### Step 2 — Switch to your branch
```bash
git checkout Kushal
git merge main   # bring the latest code into your branch
```

### Step 3 — Make 2–3 small commits

> 💡 **If you're using Claude:** Tell Claude — *"I'm working on the `Kushal` branch of a Deep Learning project at this path: [your local path]. Look at the codebase and help me make 2–3 small, meaningful commits — for example: add comments, improve a function, add a plot, fix an edge case, or add a helper script. Make each change a separate git commit with a good message."*

```bash
# After each change:
git add <changed files>
git commit -m "your descriptive message"
```

### Step 4 — Push your branch
```bash
git push origin Kushal
```

### Step 5 — Open a Pull Request
1. Go to [https://github.com/Anshul-0211/DL_project](https://github.com/Anshul-0211/DL_project)
2. You'll see a banner: **"Kushal had recent pushes"** → Click **"Compare & pull request"**
3. Title: `[Kushal] <brief description of your contributions>`
4. Write a short description of what you did
5. Set base branch to `main`
6. Click **"Create pull request"**

---

---

## 🔁 Final Workflow Summary

```
[ Ayush ]  → Push full project to main (3-4 commits) + commits on Ayush branch + PR
[ Naman ]  → Clone → switch to Naman branch → 2-3 commits → Push → PR
[ Kushal ] → Clone → switch to Kushal branch → 2-3 commits → Push → PR
[ Anshul ] → Pull main → work on Anshul branch → 2-3 commits → Push → PR → MERGE ALL PRs
```

---

## ❓ Common Issues

**Authentication error when pushing?**
```bash
git remote set-url origin https://<your-github-username>:<your-token>@github.com/Anshul-0211/DL_project.git
```
Generate a token at: https://github.com/settings/tokens (select `repo` scope)

**Merge conflict?**
Tell Claude: *"I have a merge conflict in this file. Help me resolve it."* Then share the conflicting file.

**Branch not found?**
```bash
git fetch origin
git checkout <branch-name>
```
