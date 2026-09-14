# proyecto9-js

Automatic hate speech detection in YouTube comments, using classic NLP and
machine learning. Built to help moderation teams flag hateful comments at
scale instead of reviewing every comment manually.

## Project structure

```
.specify/
├── memory/
│   └── constitution.md      # Project principles, always read first
├── templates/                # Official Spec Kit templates (spec, plan, tasks, ...)
└── scripts/bash/              # Spec Kit feature/plan/task automation scripts

specs/
├── 001-hate-speech-detection/  # Essential-level feature: spec, plan, tasks
├── 002-develop-branch-workflow/ # Spec for the develop branch setup (retroactive)
└── 003-project-structure-setup/ # Spec for this folder structure (retroactive)

src/                            # Source code implementing the specs
├── data/                        # Dataset loading (loader.py implemented)
├── preprocessing/                # Text cleaning, tokenization, stopwords, stemming/lemmatization
├── features/                      # Vectorization (TF-IDF, Bag of Words)
├── models/                         # Training and hyperparameter tuning
├── evaluation/                      # Metrics and overfitting checks
├── api/                              # Prediction service (FastAPI)
└── app/                                # User-facing interface (Streamlit)

data/
├── raw/                          # Original, unmodified dataset (gitignored)
└── processed/                     # Cleaned/preprocessed dataset (gitignored)

tests/                            # Unit tests (pytest)
notebooks/                        # Exploratory analysis notebooks
docker/                           # Container setup
.github/workflows/ci.yml          # CI: runs the test suite on every push/PR
```

## Getting started

```bash
pip install -r requirements.txt
```

See [`specs/001-hate-speech-detection/spec.md`](specs/001-hate-speech-detection/spec.md)
for the current feature scope and [`.specify/memory/constitution.md`](.specify/memory/constitution.md)
for the project's guiding principles.

## Running tests

```bash
pytest -v
```

The same suite runs automatically on every push/PR to `main` or `develop`
via GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Project management

Progress is tracked on a Kanban board (GitHub Projects):
[NLP Hate Speech Detection - Project](https://github.com/orgs/Bootcamp-IA-MAD-P7/projects/49),
with columns Backlog / Todo / In Progress / Done, mirroring the delivery
levels from the briefing (Esencial, Medio, Avanzado, Experto).

## Spec-Driven Development

This project follows Spec-Driven Development using the official
[Spec Kit](https://github.com/github/spec-kit) CLI (`specify`). Every task
gets its own spec under `specs/<issue-number>-<slug>/` before implementation
begins. If you have Claude Code installed, the bundled skills are available
after cloning:

```
/speckit-constitution   # Establish/update project principles
/speckit-specify        # Create a spec for a new task
/speckit-plan           # Create its implementation plan
/speckit-tasks          # Break it into atomic tasks
/speckit-implement      # Execute the implementation
```

## Branching workflow

- `main`: stable, finished increments only.
- `develop`: integration branch for ongoing work.
- Feature branches are created from `develop` and merged back via pull request.
