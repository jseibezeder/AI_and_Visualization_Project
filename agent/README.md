# Agent working files

Agent-only helper scripts, validation reports, run logs, source dumps, backups and
handoff notes are collected here, separate from the scientific notebooks, model
runtime code and project documentation.

- `progress.md`: concise current handoff and validation status.
- `runs/<task>/`: local agent working files from dataset setup, model updates,
  analysis development, finalization and repository cleanup. Ignored by Git.
- `runs/relocations.json`: exact original/new locations and SHA-256 hashes for
  the collected agent files.

Historical helpers can contain machine-specific paths and destructive notebook
rewrites intended for a particular development stage. They are records, **not a
supported pipeline to rerun**. In particular, do not rerun the one-time relocation
scripts against an already reorganized checkout.

The supported reproducible workflow is the three root notebooks described in
`../README.md`; `../RESULTS.md` is a project results report, not an agent log.
Scientific figures, HTML previews and intermediate arrays previously uploaded to
web chat are left in place so their existing preview links remain available.
Global Pi configuration and unrelated projects are not moved.
