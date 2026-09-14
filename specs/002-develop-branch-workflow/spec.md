# Spec: Develop branch and git branching workflow

## Linked issue

- GitHub issue: #2
- Kanban status at spec creation time: Done (written retroactively, per
  project decision to require a spec for every task, including completed
  ones)
- Level: Project Management

## What

Set up `develop` as the integration branch of the repository, created from
`main` and pushed to `origin`, so that ongoing work merges there first.

## Why

The project needs a clean git history where `main` only ever receives
finished, working increments (constitution principle: "Clean version
control"). Without a dedicated integration branch, every feature branch
would target `main` directly, making it hard to keep `main` stable and to
review work before it ships.

## Scope

- Create `develop` from `main`.
- Push `develop` to `origin` with tracking configured.

## Out of scope

- Branch protection rules on `develop`/`main` (not configured yet).
- The actual feature branches that merge into `develop` (each of those is
  its own task/spec).

## Success criteria (EARS-style)

- WHEN a contributor runs `git checkout develop`, THE SYSTEM SHALL provide a
  branch that exists both locally and on `origin`, based on `main`.
- WHEN a feature branch is ready, THE SYSTEM SHALL allow it to be merged into
  `develop` via a pull request, never directly into `main`.
- WHEN `develop` is stable and a delivery level is complete, THE SYSTEM SHALL
  allow `develop` to be merged into `main`.

## Open questions

- None at spec creation time.
