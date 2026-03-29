# llm-client Release Automation

The package release path uses two automation layers:

## Continuous Package Validation

The package CI workflow validates:

- standalone installation across supported Python versions
- live cookbook entrypoint validation
- wheel and sdist build verification
- package metadata and docs inventory tests

## Publish Automation

The publish workflow is designed for:

- tagged releases for PyPI
- manual releases for TestPyPI or PyPI

## Release Flow

1. Update version and release notes.
2. Push the version tag or trigger the manual publish workflow.
3. Build wheel and sdist.
4. Run `twine check`.
5. Publish artifacts through the package publish workflow.

## Safety Notes

- Tagged releases should publish to the primary package repository.
- Manual releases should default to TestPyPI unless there is an explicit
  release decision to publish to PyPI.
