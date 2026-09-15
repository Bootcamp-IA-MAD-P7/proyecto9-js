# Phase 0 Research: Classic text vectorization

No `[NEEDS CLARIFICATION]` markers remained in the spec, but the choices
below were still worth writing down explicitly since they shape
`src/features/vectorize.py`'s API.

## Decision: scikit-learn `TfidfVectorizer` / `CountVectorizer`

**Rationale**: already the project's target ML stack (per
`specs/001-hate-speech-detection/tasks.md` sections 4-6: Logistic
Regression, Linear SVM, Naive Bayes, Optuna tuning — all scikit-learn
native). Both vectorizers share the same `analyzer`/`token_pattern`/
`min_df` arguments, so swapping one for the other in
`fit_transform_corpus()` is a one-line change, keeping the comparison
in `scripts/build_features.py` apples-to-apples.

**Alternatives considered**:
- `gensim` `Dictionary`/`TfidfModel`: more common for topic modeling
  pipelines; would need a bridge layer to hand scikit-learn estimators a
  numpy/scipy matrix, adding a dependency with no benefit over
  scikit-learn's native, sparse-matrix-returning vectorizers.
- Hand-rolled counting (`collections.Counter` + manual IDF math): no
  meaningful control benefit over scikit-learn's battle-tested
  implementation, and loses `min_df`/`max_df` vocabulary pruning for free.

## Decision: fit on already-preprocessed `clean_comment`, not raw `comment`

**Rationale**: `clean_comment` (from
`specs/006-text-preprocessing/spec.md`) is already lowercased, URL/
mention/HTML-stripped, tokenized, stopword-free, and stemmed per-language.
Vectorizing the raw `comment` column instead would re-introduce noise
(URLs, punctuation, stopwords) that preprocessing was specifically built
to remove, and would double-count work.

**Alternatives considered**: vectorizing `comment` directly with
scikit-learn's own built-in tokenizer/stopword list — rejected because
scikit-learn's default English-only stopword list has no Spanish
equivalent, and the corpus is ~51% Spanish.

## Decision: `min_df=5` shared vocabulary-pruning threshold

**Rationale**: with 151,848 rows, singleton/near-singleton tokens (typos,
usernames that survived cleaning, transliteration artifacts) inflate
vocabulary size without adding generalizable signal, and bloat the sparse
matrix's column count for no benefit to a linear classifier. `min_df=5`
is a light, standard-practice floor (drop tokens appearing in fewer than
5 documents) — conservative enough not to drop legitimate rare slurs that
matter for the minority hateful class.

**Alternatives considered**: no pruning (`min_df=1`) — rejected, produces
a needlessly large vocabulary dominated by long-tail noise, which the
Result section's actual vocabulary-size numbers will confirm or correct
after the script runs. `max_features` cap instead of `min_df` — rejected
as a first pass, since an arbitrary top-N cap could silently drop rare
but category-relevant terms (e.g. `classism`-specific vocabulary, already
critically thin per `specs/048-spanish-category-augmentation/spec.md`);
`min_df` prunes noise without an arbitrary ceiling.

## Decision: persist fitted vectorizers with `joblib`

**Rationale**: scikit-learn's own documented recommendation for
persisting fitted estimators (handles numpy arrays inside the object more
efficiently than raw `pickle`), and needed per
`specs/001-hate-speech-detection/tasks.md` section 4 ("persist it
alongside its fitted vectorizer for reuse at inference time").

**Alternatives considered**: raw `pickle` — works, but `joblib` is the
scikit-learn-recommended, already-transitively-available choice (bundled
with scikit-learn), so no new dependency is introduced.
