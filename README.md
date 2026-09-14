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

Download the dataset from the briefing's
[Google Drive link](https://drive.google.com/file/d/1bG7fA273jIBgJfc6YS1vsKfr1qRiNUTU/view)
(`youtoxic_english_1000.csv`) and place it under `data/raw/` (gitignored —
data is not versioned). `src/data/loader.py` maps its native
`Text`/`IsHatespeech` columns to this project's `comment`/`label` schema.

### Enriched multi-category dataset (optional)

The briefing dataset alone only has a single hate/not-hate flag and just
1,000 rows. To combine it with six additional public datasets (HateXplain,
ETHOS, Measuring Hate Speech, HatEval, HaterNet, OffendES) into a richer,
bilingual dataset with category labels (racism, xenophobia, religion, misogyny,
homophobia, transphobia, disability, classism, violence):

1. Download the freely-available sources (no account needed):

   ```bash
   mkdir -p data/raw/external
   curl -L "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json" -o data/raw/external/hatexplain_dataset.json
   curl -L "https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv" -o data/raw/external/ethos_binary.csv
   curl -L "https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Multi_Label.csv" -o data/raw/external/ethos_multilabel.csv
   curl -L "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/main/measuring-hate-speech.parquet" -o data/raw/external/measuring_hate_speech.parquet
   curl -L "https://zenodo.org/records/2592149/files/labeled_corpus_6K.txt" -o data/raw/external/haternet_labeled_corpus_6k.txt
   mkdir -p data/raw/external/offendes
   curl -L "https://raw.githubusercontent.com/fmplaza/OffendES/main/split_MeOffendES/training_set.tsv" -o data/raw/external/offendes/training_set.tsv
   curl -L "https://raw.githubusercontent.com/fmplaza/OffendES/main/split_MeOffendES/dev_set.tsv" -o data/raw/external/offendes/dev_set.tsv
   curl -L "https://raw.githubusercontent.com/fmplaza/OffendES/main/split_MeOffendES/test_set.tsv" -o data/raw/external/offendes/test_set.tsv
   ```

2. Download HatEval (the Spanish-language source) manually: create a free
   [HuggingFace account](https://huggingface.co/join), accept the access
   gate on [valeriobasile/HatEval](https://huggingface.co/datasets/valeriobasile/HatEval),
   then download `train-00000-of-00001.parquet`, `dev-00000-of-00001.parquet`,
   and `test-00000-of-00001.parquet` from its "Files and versions" tab into
   `data/raw/external/hateval/` as `train.parquet`, `dev.parquet`, and
   `test.parquet`.

3. Build the combined dataset:

   ```bash
   python scripts/build_enriched_dataset.py
   ```

This writes `data/processed/enriched_comments.csv` (~117,700 rows, ~29%
hate, ~43,000 Spanish rows — over a third of the dataset). See
[`specs/004-dataset-enrichment/spec.md`](specs/004-dataset-enrichment/spec.md)
for the category-mapping decisions.

### Auxiliary sarcasm dataset (optional, English + machine-translated Spanish)

Kept separate from the hate-speech dataset above (different corpus, different
annotation dimension — see
[`specs/005-sarcasm-dataset/spec.md`](specs/005-sarcasm-dataset/spec.md) for
why). To build it:

1. Download the English source (no account needed):

   ```bash
   curl -L "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json" -o data/raw/external/sarcasm_headlines_en.json
   ```

2. Install the heavy NLP dependencies (only needed for translation):

   ```bash
   pip install -r requirements-nlp.txt
   ```

3. Build the combined English + Spanish dataset:

   ```bash
   python scripts/build_sarcasm_dataset.py
   ```

This writes `data/processed/sarcasm_comments.csv` (~57,200 rows). The
Spanish half is machine-translated (local model, no rate limits), not
native Spanish sarcasm — translation takes ~80-90 minutes on CPU for the
full corpus.

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
- Both `main` and `develop` are protected: the CI check must pass before a
  PR can be merged, and neither branch accepts force-pushes or deletion.
