# proyecto9-js

Automatic hate speech detection in YouTube comments, using classic NLP and
machine learning. Built to help moderation teams flag hateful comments at
scale instead of reviewing every comment manually.

## Project structure

```
.specify/
└── memory/
    └── constitution.md      # Project principles, always read first

specs/
└── 001-hate-speech-detection/
    ├── spec.md               # What and why
    ├── plan.md                # Architecture and technical decisions
    └── tasks.md                # Atomic tasks with EARS-style criteria

src/                            # Source code implementing the specs
├── data/                        # Dataset loading
├── preprocessing/                # Text cleaning, tokenization, stopwords, stemming/lemmatization
├── features/                      # Vectorization (TF-IDF, Bag of Words)
├── models/                         # Training and hyperparameter tuning
├── evaluation/                      # Metrics and overfitting checks
├── api/                              # Prediction service (FastAPI)
└── app/                                # User-facing interface (Streamlit)

data/
├── raw/                          # Original, unmodified dataset
└── processed/                     # Cleaned/preprocessed dataset

tests/                            # Unit tests
notebooks/                        # Exploratory analysis notebooks
docker/                           # Container setup
```

## Getting started

```bash
pip install -r requirements.txt
```

See [`specs/001-hate-speech-detection/spec.md`](specs/001-hate-speech-detection/spec.md)
for the current feature scope and [`.specify/memory/constitution.md`](.specify/memory/constitution.md)
for the project's guiding principles.

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
