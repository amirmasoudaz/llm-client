# Agent Version Control Policy

Last updated: 2026-04-02

This document defines the required version-control workflow for AI agents and
human contributors working on this repository.

The goal is simple:

- keep work professional
- keep changes traceable
- keep `main` releasable
- ensure every feature, fix, and release step is reviewable

This repository follows a **PR-gated trunk workflow** with:

- short-lived branches
- pull-request / merge-request based integration
- semantic versioning
- release branches for version bumps and release prep

It is closest to GitHub Flow, but extended with explicit release-branch
discipline.

## Non-Negotiable Rules

- Never commit directly to `main`.
- Every logical change gets its own branch.
- Every branch merges through a PR or MR.
- Tags are created only from merged commits on `main`.
- Do not mix unrelated work into one branch.
- Public behavior changes must update docs in the same branch.
- Validation results must be reported honestly. Do not hide failures.

## Source Of Truth

- `main` is the source of truth.
- `main` should remain releasable.
- Feature branches are temporary working branches.
- Release tags mark shipped versions.

## Branch Naming Rules

Use short, scoped branch names.

- `feat/...` for new features
- `fix/...` for bug fixes
- `refactor/...` for internal code improvements
- `chore/...` for maintenance, metadata, tooling, or dependency cleanup
- `docs/...` for documentation-only changes
- `release/x.y.z` for release preparation and version bumps

Examples:

- `feat/openai-realtime-support`
- `fix/files-api-purpose-download`
- `refactor/optional-cache-deps`
- `chore/fix-project-urls`
- `docs/agent-version-control-policy`
- `release/1.1.1`

## One Branch, One Purpose

Each branch must contain one coherent unit of work.

Good examples:

- a metadata fix only
- one feature implementation only
- one refactor only
- one release-preparation change only

Bad examples:

- feature implementation plus unrelated docs cleanup
- dependency refactor plus random example edits
- release bump plus unrelated experimental work

## Required Workflow For Every Change

1. Start from `main`.
2. Create a new branch with a scoped name.
3. Make the smallest coherent change that fully addresses the task.
4. Run focused validation.
5. Update docs/examples if public behavior changed.
6. Commit with a clear, direct message.
7. Push the branch.
8. Open a PR or MR.
9. Merge only after review/checks.
10. Delete the branch after merge.

## Commit Policy

Commits must be:

- scoped
- readable
- reversible
- meaningful without extra explanation

Good commit messages:

- `Fix project metadata URLs`
- `Make cache backend deps optional`
- `Prepare 1.1.1 patch release`
- `Add OpenAI background response helpers`
- `Handle assistants-purpose files without content download`

Optional stricter style is also acceptable:

- `feat: add OpenAI background response helpers`
- `fix: handle assistants-purpose files without content download`
- `refactor: lazy-load pg_redis cache dependencies`
- `docs: add agent version control policy`
- `chore: prepare 1.1.1 patch release`

## PR / MR Policy

All integration happens through PRs or MRs.

Each PR should clearly answer:

- What changed?
- Why did it change?
- How was it validated?
- What are the risks?
- Were docs/examples/versioning updated if needed?

Minimum PR content:

- short summary
- scope boundaries
- validation notes
- migration/install notes if relevant

## Validation Policy

Validation must match the scope of the change.

Examples:

- metadata fix: sanity-check file diff and metadata correctness
- runtime fix: focused tests for the changed code path
- refactor: import-safety and regression tests
- release branch: version check, release notes, smoke or cookbook validation

Agents must report:

- which tests were run
- whether they passed
- what was not run
- any environment-related skips or blockers

## Documentation Policy

If a public behavior changes, update documentation in the same branch.

This may include:

- package README
- API guides
- usage guides
- examples
- release notes
- audits or evaluation reports

Do not defer docs for shipped behavior.

## Release Policy

Version bumps and release prep must happen on a dedicated release branch.

For a patch release:

1. branch from `main` using `release/x.y.z`
2. cherry-pick or merge the intended fixes into that release branch
3. bump the package version
4. add or update release notes
5. run release validation
6. open a PR from the release branch to `main`
7. merge the PR
8. create the tag from the merged `main` commit
9. push the tag

Important:

- do not tag a feature branch
- do not tag an unmerged release branch
- tags must point to the merged commit on `main`

## Tagging Policy

Tag format:

- `v1.1.1`
- `v2.0.0`

Create tags only after merge to `main`.

Example:

```bash
git checkout main
git pull origin main
git tag -a v1.1.1 -m "llm-client 1.1.1"
git push origin v1.1.1
```

## Dirty Working Tree Policy

Do not mix unrelated local files into a branch.

If a file is unrelated to the current task:

- leave it untracked
- leave it unstaged
- do not include it in the PR

Never commit mystery files “just because they exist.”

## Rules For AI Agents

Every AI agent working on this repository must:

- check branch and working tree state first
- branch before making scoped changes
- avoid direct work on `main`
- keep changes limited to the stated task
- update docs when public behavior changes
- run focused validation
- commit clearly
- push a branch, not `main`
- stop at PR-ready unless explicitly asked to merge or release
- never create a release tag from an unmerged branch
- never hide validation failures
- never bundle unrelated work into a single PR

## Standard Agent Checklist

Use this checklist for every task:

1. Check current branch and working tree.
2. If on `main`, create a scoped branch.
3. State task scope in one sentence.
4. Edit only relevant files.
5. Run focused validation.
6. Update docs/examples if needed.
7. Commit with a clear message.
8. Push the branch.
9. Report:
   - branch name
   - commit hash
   - validation run
   - PR link if created
10. If the task is a release:
   - use `release/x.y.z`
   - merge via PR
   - tag only after merge

## Recommended PR Order For Related Work

When multiple related branches exist, merge in dependency order.

Example:

1. `chore/...`
2. `refactor/...`
3. `feat/...`
4. `release/x.y.z`

This keeps the release branch small and easier to review.

## What To Call This Workflow

Use one of these names internally:

- `PR-Gated Trunk Workflow`
- `Short-Lived Branch + Release Branch Workflow`

Preferred term for this repository:

**PR-Gated Trunk Workflow**
