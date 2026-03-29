# llm-client Changelog Process

`llm-client` uses a repository-maintained changelog process tied to versioned
package releases.

## Rules

- Every released version must have a changelog entry.
- Breaking changes must be called out explicitly.
- Compatibility-only changes and deprecations must be grouped separately from
  new features.
- Changes to the stable API map, installation matrix, or support policy must be
  mentioned in the release notes.

## Suggested Changelog Sections

- Added
- Changed
- Deprecated
- Fixed
- Security

## Release Note Inputs

Each release should summarize:

- provider/runtime changes
- agent/tool/context/memory changes
- packaging or installation changes
- docs/examples/benchmark changes
- known limitations or follow-up items

## Source of Truth

The changelog can live in repository release notes or a dedicated `CHANGELOG.md`
once that file is introduced. The requirement for this package phase is the
process itself: no package release should ship without human-curated notes.
