# Contributing to DL_Project

Thank you for contributing! This document explains the team workflow, branching strategy, and code guidelines.

---

## 🌿 Branching Strategy

Each team member works on their **own named branch** and merges into `main` via a Pull Request.

```
main          ← stable, production-ready code
 ├── Anshul   ← Anshul's contributions
 ├── Ayush    ← Ayush's contributions
 ├── Naman    ← Naman's contributions
 └── Kushal   ← Kushal's contributions
```

### Rules
- **Never commit directly to `main`** — always work on your branch
- Always **pull the latest `main`** before starting new work
- Open a **Pull Request** when your work is ready to merge
- PRs are reviewed and merged by the repo owner (Anshul)

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/Anshul-0211/DL_project.git
cd DL_project

# 2. Switch to your branch
git checkout <YourName>

# 3. Pull latest changes from main into your branch
git merge main
```

---

## 💾 Making Commits

Make small, focused commits with clear messages:

```bash
git add <files>
git commit -m "type(scope): short description"
```

### Commit Message Types
| Type | When to use |
|------|-------------|
| `feat` | Adding a new feature or function |
| `fix` | Bug fix |
| `docs` | Documentation or docstring changes |
| `refactor` | Code restructuring without behaviour change |
| `test` | Adding or updating tests |
| `chore` | Config, dependencies, tooling |

**Examples:**
```
feat(metrics): add hausdorff_distance utility
fix(pipeline): handle empty landmark array edge case
docs(geometry): add docstrings to normalize_mesh
```

---

## 🔃 Opening a Pull Request

1. Push your branch: `git push origin <YourName>`
2. Go to [https://github.com/Anshul-0211/DL_project](https://github.com/Anshul-0211/DL_project)
3. Click **"Compare & pull request"**
4. Set **base branch** to `main`
5. Write a short description of your changes
6. Click **"Create pull request"**

---

## 🧪 Running Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

Make sure all tests pass before opening a PR.

---

## 🐍 Code Style

- Follow **PEP 8** conventions
- Add **docstrings** to all public functions
- Use **type hints** where possible
- Keep functions **focused and small** — one responsibility per function

---

## 📁 Project Structure

```
DL_project/
├── face_semantic_icp/   # Core package — models, geometry, metrics, pipeline
├── configs/             # Training and dataset configuration files
├── scripts/             # Standalone helper and visualisation scripts
├── tests/               # Unit tests
├── outputs/             # Generated results and evaluation reports
└── showcase_2026_07_04/ # End-to-end showcase run artifacts
```

---

*For questions, reach out to Anshul (repo owner).*
