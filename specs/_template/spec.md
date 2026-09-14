# Spec: {{Task title}}

> Base template. Copy this file into `specs/<issue-number>-<short-slug>/spec.md`
> for every new task/issue before implementation begins, then fill in each
> section. Delete this instructions block once adapted.

## Linked issue

- GitHub issue: #{{issue-number}}
- Kanban status at spec creation time: {{Backlog / Todo / In Progress / Done}}
- Level: {{Nivel Esencial / Medio / Avanzado / Experto / Project Management}}

## What

One or two sentences: what does this task actually build or change?

## Why

Why does this task matter for the project's goal (automatically flagging
hate speech in YouTube comments)? Tie it back to the briefing or to a
principle in [`.specify/memory/constitution.md`](../../.specify/memory/constitution.md)
when relevant.

## Scope

- Bullet list of what this specific task covers.

## Out of scope

- Bullet list of what it explicitly does not cover (usually: other tasks/
  issues in the Kanban board).

## Success criteria (EARS-style)

Write each criterion as one of:

- `WHEN <trigger>, THE SYSTEM SHALL <observable behavior>.`
- `IF <condition>, THEN THE SYSTEM SHALL <observable behavior>.`
- `WHERE <context>, THE SYSTEM SHALL <observable behavior>.`

These sentences must be specific enough to become test assertions later
(see `tests/`), not vague goals like "it should work well".

- WHEN ..., THE SYSTEM SHALL ...

## Open questions

- Anything left undecided that implementation will need to resolve.
