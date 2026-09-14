# Project Constitution

Core principles guiding this project. This file is read before any specification
or implementation work.

## Context

YouTube needs to automatically detect hate speech in video comments, since manual
moderation cannot scale with platform growth. The client values a **practical,
deployable solution over a maximally precise one**.

## Principles

1. **Practicality over precision.** A working, usable solution beats a marginally
   more accurate model that is hard to ship or maintain.
2. **Generalization over memorization.** Overfitting must be actively controlled;
   the gap between training and test metrics must stay under 5 percentage points.
3. **Reproducibility.** Preprocessing, training, and evaluation must be scriptable
   and repeatable, not exploratory notebook-only work.
4. **Incremental delivery.** Features are scoped and delivered level by level
   (essential → medium → advanced → expert), each one shippable on its own.
5. **Clean version control.** Well-organized branches, descriptive commits, and
   pull requests reviewed before merging into `develop`, and from `develop` into
   `main` only when a level is complete and stable.
6. **Documented code.** Every module, script, and API endpoint must be
   understandable without needing to ask the author.

## Workflow

- All work happens on feature branches created from `develop`.
- `develop` is the integration branch; `main` only receives finished, working
  increments.
- Every task/issue has its own spec under `specs/<issue-number>-<slug>/spec.md`
  before implementation begins in `src/`, adapted from the base template at
  [`specs/_template/spec.md`](../../specs/_template/spec.md). This applies to
  new tasks going forward, and was applied retroactively to tasks that were
  already Done when this rule was adopted.
- Branch names, commit messages, and code comments are written in English.
