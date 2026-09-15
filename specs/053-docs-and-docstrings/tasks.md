# Tasks: Document code and write project README

**Input**: Design documents from `/specs/053-docs-and-docstrings/`

**Prerequisites**: plan.md, spec.md, quickstart.md

**Tests**: none new — documentation-only change; the existing suite
must still pass unmodified.

**Organization**: single cohesive story (US1) — audit, fix docstrings,
update README.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [x] T001 Create `specs/053-docs-and-docstrings/` spec, plan,
      quickstart (this session, prior steps)

## Phase 2: Foundational

None needed — this task only edits existing files.

## Phase 3: User Story 1 - Every module documented, README accurate and results-visible (Priority: P1) 🎯 MVP

**Goal**: no public function in `src/` lacks a docstring, the README's
project-structure description matches reality, and the final Nivel
Esencial results are visible without scrolling past every pipeline
subsection.

**Independent Test**: run the audit script from `quickstart.md` and
confirm it returns `[]`; read the README's "Project structure" section
and confirm every listed directory/file matches the actual repo tree;
confirm the new "Results at a glance" section states the final chosen
model and its headline metrics.

### Implementation for User Story 1

- [x] T002 [P] [US1] Run the AST docstring audit from `quickstart.md`
      against `src/`, confirming the 3 missing docstrings named in
      spec.md's Scope (no other files affected)
- [x] T003 [US1] Add a one-line docstring to `src/api/main.py::health`
      and `src/api/main.py::predict` (depends on T002 confirming these
      are the only gaps in `src/api/`)
- [x] T004 [US1] Add a one-line docstring to
      `src/app/streamlit_app.py::get_model_and_vectorizer` (depends on
      T002)
- [x] T005 [US1] Re-run the audit script, confirm it now returns `[]`
      (depends on T003, T004)
- [x] T006 [US1] Rewrite the README's "Project structure" section:
      replace the stale `specs/` enumeration (3 of 13 directories) with
      a pattern-level description, and correct the `src/` package
      descriptions to match current contents (data/, preprocessing/,
      features/, models/, evaluation/, api/, app/ all populated)
- [x] T007 [US1] Add a "Results at a glance" section to the README
      (placed early, before the detailed pipeline walkthrough) stating:
      final dataset size/composition, chosen baseline model, test-split
      accuracy/precision/recall/F1, and the overfit verdict, each linking
      to its source spec for detail

**Checkpoint**: User Story 1 (the whole feature) is complete and
independently verifiable.

## Phase 4: Polish & Cross-Cutting Concerns

- [x] T008 [P] Update `specs/001-hate-speech-detection/tasks.md` section 8
      checkbox to `[x]`
- [x] T009 Run `pytest` (full suite) and confirm no regressions
      (documentation-only change — this should be a no-op verification)
- [x] T010 Run `quickstart.md`'s audit command one final time as the
      closing verification

## Dependencies & Execution Order

- Setup (T001) → done already, no blockers
- Foundational: none needed, US1 can start immediately
- Within US1: T002 (audit) before T003/T004 (fixes) before T005
  (re-audit); T006/T007 (README) are independent of T002-T005 and of
  each other
- Polish (T008-T010) depends on US1 (T002-T007) being complete

## Implementation Strategy

### MVP First (User Story 1 Only)

1. T002: confirm the audit's 3 findings match spec.md
2. T003, T004: add the missing docstrings
3. T005: re-run the audit, confirm `[]`
4. T006, T007: fix the README
5. **STOP and VALIDATE**: audit returns `[]`, README reads accurately

### Incremental Delivery

This feature *is* the MVP — there is only one story. Polish (T008-T010)
wraps it up: update the master tasks.md, run the full test suite to
prove nothing behavioral changed, and do a final quickstart pass before
opening the PR.
