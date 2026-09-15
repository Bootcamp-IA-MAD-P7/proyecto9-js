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
1,000 rows. To combine it with eight additional public datasets (HateXplain,
ETHOS, Measuring Hate Speech, HatEval, HaterNet, OffendES, HaSCoSVa,
DETESTS) into a richer, bilingual dataset with category labels (racism,
xenophobia, religion, misogyny, homophobia, transphobia, disability,
classism, violence):

1. Download the freely-available sources (no account needed):

   ```bash
   mkdir -p data/raw/external
   curl -L "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json" -o data/raw/external/hatexplain_dataset.json
   curl -L "https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv" -o data/raw/external/ethos_binary.csv
   curl -L "https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Multi_Label.csv" -o data/raw/external/ethos_multilabel.csv
   curl -L "https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/main/measuring-hate-speech.parquet" -o data/raw/external/measuring_hate_speech.parquet
   curl -L "https://zenodo.org/records/2592149/files/labeled_corpus_6K.txt" -o data/raw/external/haternet_labeled_corpus_6k.txt
   curl -L "https://gitlab.inria.fr/counter/HaSCoSVa/-/raw/main/dataset/hascosva_2022_anonymized.tsv" -o data/raw/external/hascosva.tsv
   mkdir -p data/raw/external/offendes
   curl -L "https://raw.githubusercontent.com/fmplaza/OffendES/main/split_MeOffendES/training_set.tsv" -o data/raw/external/offendes/training_set.tsv
   curl -L "https://raw.githubusercontent.com/fmplaza/OffendES/main/split_MeOffendES/dev_set.tsv" -o data/raw/external/offendes/dev_set.tsv
   curl -L "https://raw.githubusercontent.com/fmplaza/OffendES/main/split_MeOffendES/test_set.tsv" -o data/raw/external/offendes/test_set.tsv
   ```

2. Download HatEval and DETESTS manually — both need a free
   [HuggingFace account](https://huggingface.co/join) and a click-through
   access gate (no personal-information form, just "agree and access"):
   - [valeriobasile/HatEval](https://huggingface.co/datasets/valeriobasile/HatEval):
     download `train-00000-of-00001.parquet`, `dev-00000-of-00001.parquet`,
     and `test-00000-of-00001.parquet` from "Files and versions" into
     `data/raw/external/hateval/` as `train.parquet`, `dev.parquet`, and
     `test.parquet`.
   - [CLiC-UB/DETESTS-Dis](https://huggingface.co/datasets/CLiC-UB/DETESTS-Dis):
     download `train.csv` and `test.csv` from "Files and versions" into
     `data/raw/external/detests/`.

3. Build the combined dataset:

   ```bash
   python scripts/build_enriched_dataset.py
   ```

This writes `data/processed/enriched_comments.csv` (~133,800 rows, ~29%
hate, ~59,100 Spanish rows — near-parity with English). See
[`specs/004-dataset-enrichment/spec.md`](specs/004-dataset-enrichment/spec.md)
for the category-mapping decisions.

### Spanish augmentation for zero-coverage categories (optional)

Seven categories (homophobia, racism, violence, religion, disability,
transphobia, classism) have zero or near-zero native Spanish rows in the
enriched dataset. This machine-translates the existing English hateful
comments in those categories into Spanish (local model, no rate limits),
tagging translated rows with a `_es_mt` source suffix so they stay
distinguishable from native Spanish data. See
[`specs/048-spanish-category-augmentation/spec.md`](specs/048-spanish-category-augmentation/spec.md)
for the rationale and the public-dataset alternatives considered.

1. Install the heavy NLP dependencies (only needed for translation), if not
   already installed for the sarcasm dataset above:

   ```bash
   pip install -r requirements-nlp.txt
   ```

2. Build the augmented dataset:

   ```bash
   python scripts/build_spanish_augmentation.py
   ```

This writes `data/processed/enriched_comments_es_augmented.csv` (the
enriched dataset plus translated Spanish rows), taking roughly the same
order of magnitude as the sarcasm dataset run (~80-90 minutes on CPU). The
full run translated 18,040 rows, growing the dataset from 133,808 to
151,848 rows (77,166 Spanish / 74,682 English) — see
[`specs/048-spanish-category-augmentation/spec.md`](specs/048-spanish-category-augmentation/spec.md#result)
for the full before/after comparison.

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

### Text preprocessing

Language-aware cleaning, tokenization, stopword removal, and stemming
(English + Spanish) — see
[`specs/006-text-preprocessing/spec.md`](specs/006-text-preprocessing/spec.md).
To apply it to the enriched dataset:

```bash
python scripts/preprocess_enriched_dataset.py
```

This writes `data/processed/enriched_comments_preprocessed.csv` (adds a
`clean_comment` column), taking ~45 seconds for the full 133,808 rows. Pass
`--input`/`--output` to run it against another harmonized CSV, e.g. the
Spanish-augmented dataset:

```bash
python scripts/preprocess_enriched_dataset.py \
  --input data/processed/enriched_comments_es_augmented.csv \
  --output data/processed/enriched_comments_es_augmented_preprocessed.csv
```

### Exploratory Data Analysis (EDA)

[`notebooks/eda_enriched_dataset.ipynb`](notebooks/eda_enriched_dataset.ipynb)
is committed **already executed** — open it on GitHub to see every plot
and finding without running anything. See
[`specs/007-eda-enriched-dataset/spec.md`](specs/007-eda-enriched-dataset/spec.md)
for the full write-up. To regenerate it after changing the data or the
pipeline:

```bash
python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('notebooks/eda_enriched_dataset.ipynb', as_version=4)
NotebookClient(nb, timeout=600, resources={'metadata': {'path': 'notebooks'}}).execute()
nbformat.write(nb, 'notebooks/eda_enriched_dataset.ipynb')
"
```

[`notebooks/eda_enriched_dataset_es_augmented.ipynb`](notebooks/eda_enriched_dataset_es_augmented.ipynb)
re-runs the same analysis on the Spanish-augmented dataset, to see how much
each finding shifts once the 7 orphan categories have Spanish coverage —
same sections, same plot types, findings grounded in the new numbers.

### Feature extraction (TF-IDF / Bag of Words)

Vectorizes the `clean_comment` column with both TF-IDF and Bag of Words,
compares their feature-space size, and persists the TF-IDF baseline for
reuse at inference time — see
[`specs/049-classic-text-vectorization/spec.md`](specs/049-classic-text-vectorization/spec.md).

```bash
python scripts/build_features.py
```

This writes `data/processed/tfidf_vectorizer.joblib`. On the full 151,848
row Spanish-augmented dataset both vectorizers share an 18,417-token
vocabulary (`min_df=5`, 99.94% sparse).

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
[proyecto9-js - NLP Hate Speech Detection](https://github.com/orgs/Bootcamp-IA-MAD-P7/projects/50),
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
