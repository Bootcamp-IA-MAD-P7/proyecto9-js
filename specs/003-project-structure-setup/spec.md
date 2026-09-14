# Spec: Initial project structure and specs scaffolding

## Linked issue

- GitHub issue: #3
- Kanban status at spec creation time: Done (written retroactively, per
  project decision to require a spec for every task, including completed
  ones)
- Level: Project Management

## What

Create the project's base folder structure: `.specify/memory/constitution.md`,
the `specs/` directory (spec-driven development scaffolding), the `src/`
package layout, and supporting directories (`data/`, `tests/`, `notebooks/`,
`docker/`), plus `requirements.txt`, `.gitignore`, and an updated `README.md`.

## Why

Before any ML code is written, the project needs an agreed-upon place for
everything (constitution, specs, source code, tests, data, docs) so that
Spec-Driven Development can actually be followed: every future task should
know exactly which folder its spec, code, and tests belong in.

## Scope

- `.specify/memory/constitution.md` with the project's guiding principles.
- `specs/001-hate-speech-detection/{spec,plan,tasks}.md` for the essential
  delivery level.
- `src/{data,preprocessing,features,models,evaluation,api,app}/` package
  layout with `__init__.py` files.
- `data/{raw,processed}/`, `tests/`, `notebooks/`, `docker/` directories.
- `requirements.txt`, `.gitignore`, and `README.md` documenting the
  structure and branching workflow.

## Out of scope

- Any actual implementation code (that belongs to each individual task's own
  spec, e.g. `specs/004-.../spec.md` for dataset loading).
- CI/testing harness (covered by a later task, see the SDD/harness work on
  branch `docs/sdd-harness-guide`).

## Success criteria (EARS-style)

- WHEN a new contributor clones the repository, THE SYSTEM SHALL present a
  `specs/` folder documenting what is being built and a `src/` folder
  mirroring the architecture described in `plan.md`.
- WHEN a new feature is started, THE SYSTEM SHALL have a place for it under
  `specs/<number>-<slug>/` following the same three-file pattern
  (`spec.md`, `plan.md`, `tasks.md`).

## Open questions

- None at spec creation time.
