# Spec 053: Document code and write project README

## Linked issue

- GitHub issue: #10
- Kanban status at spec creation time: In Progress
- Level: Nivel Esencial

## What

Close out the Nivel Esencial milestone's documentation task: audit every
`src/`/`scripts/` module for docstrings, add the handful that are
missing, fix the now-stale "Project structure" section of the README
(written back in `specs/003-project-structure-setup/spec.md`, before any
of `src/data/`, `src/features/`, `src/models/`, `src/evaluation/`,
`src/api/`, `src/app/` had real content), and add a concise
results-at-a-glance summary so the final Nivel Esencial numbers don't
require scrolling through every pipeline subsection to find.

## Why

Every prior spec in this Nivel Esencial arc (004 through 052) already
documented its own setup/usage/results in its README subsection as it
landed — the constitution's "Documented code" principle has been applied
incrementally throughout, not deferred to the end. What's left, now that
issue #9 (the last functional task) is done, is the audit-and-cleanup
pass issue #10 exists for: catching anything that slipped through
(missing docstrings on the two newest modules, `src/api/main.py` and
`src/app/streamlit_app.py`), and fixing documentation that was accurate
when written but drifted as the codebase grew (the "Project structure"
section still describes `src/data/` as "loader.py implemented" and lists
only 3 of the 13 spec directories that exist today).

## Scope

- Add docstrings to the functions an AST audit found missing them:
  `src/api/main.py::health`, `src/api/main.py::predict`,
  `src/app/streamlit_app.py::get_model_and_vectorizer`. (Audit method:
  walk every `src/**/*.py` file's AST, flag public functions with no
  docstring — see Result section for the full audit output.)
- Rewrite the README's "Project structure" section to reflect the actual
  current tree (all populated `src/` packages, and a pattern-level
  description of `specs/` instead of a stale enumeration of 3 out of 53
  directories).
- Add a "Results at a glance" section to the README summarizing the
  final Nivel Esencial numbers (dataset size/composition, chosen model,
  headline metrics, overfit verdict) in one place, linking to the
  detailed per-task subsections and specs for the full picture.
- Explicitly NOT rewriting or restructuring the existing per-task README
  subsections (Getting started through Serving) — those are current and
  accurate as of specs 048-052 and already satisfy "document setup and
  usage."

## EARS criteria

1. WHEN any `src/**/*.py` file is parsed, THE SYSTEM SHALL have a
   module-level docstring and a docstring on every public (non-`_`
   prefixed) function or method — verified by the AST audit script, not
   by manual inspection alone.
2. THE SYSTEM SHALL describe, in the README's "Project structure"
   section, only directories and package roles that exist and are
   populated in the actual tree at spec-completion time (no stale
   "(loader.py implemented)"-style annotations that predate later work).
3. THE SYSTEM SHALL surface the final Nivel Esencial model metrics
   (chosen model, test accuracy/precision/recall/F1, overfit verdict)
   within the first screen of the README, not only inside the "Model
   evaluation" subsection several hundred lines down.

## Result

AST audit before this spec's fixes found exactly the 3 gaps anticipated:
`src/api/main.py::health`, `src/api/main.py::predict`, and
`src/app/streamlit_app.py::get_model_and_vectorizer` (empty `__init__.py`
package markers excluded — those are intentional, not undocumented
code). Added one-line docstrings to all 3; re-running the audit now
returns `[]` across all of `src/`.

README changes: rewrote the "Project structure" section (was still
describing `src/data/` as "loader.py implemented" and enumerating only
3 of the 13 real `specs/` directories) to describe the actual current
tree and point at `ls specs/` for the up-to-date list instead of an
enumeration that will go stale again. Added a "Results at a glance"
section right after the intro, summarizing the final Nivel Esencial
numbers (dataset, features, model, test metrics, overfit verdict,
serving) with links to each source spec.

Full test suite re-run after all changes: 48/48 passed, `ruff check .`
clean — confirms this was purely a documentation change with no
behavioral impact, as intended.

## Out of scope

- Sphinx/mkdocs or any generated API documentation site — issue #10 asks
  for docstrings + README, not a documentation site; no later issue
  names one either.
- Re-auditing or rewriting the per-task README subsections that already
  document their own setup/usage/results (specs 004-052) — only the
  stale "Project structure" section and the missing top-level results
  summary are in scope.
