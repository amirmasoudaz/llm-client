# llm-client Release Automation

The package release path is intentionally manual.

This repository does not rely on GitHub-hosted CI or publish workflows for the
package release path. Validation and release publication are expected to run
from a local, controlled environment.

## Validation Expectations

Before a release is merged and tagged, validate:

- standalone installation in the repository `.venv`
- package metadata and docs inventory tests
- full test suite
- cookbook validation with expected environment-dependent skips only
- wheel and sdist build verification
- `twine check` on built artifacts

## Release Flow

1. Update the package version and release notes on a release-prep branch.
2. Run the validation gate locally.
3. Merge the release branch into `main` through the normal PR flow.
4. Tag the merged `main` commit.
5. Build wheel and sdist locally if distribution artifacts are needed.
6. Run `twine check` locally before any manual publish step.
7. Publish artifacts manually only if there is an explicit release decision to
   do so.

## Safety Notes

- Do not treat tag creation as automatic publication.
- Prefer validating and tagging from a clean local checkout of merged `main`.
- If artifacts are published, default to the lowest-risk target first unless
  there is an explicit decision to publish to the primary package index.
