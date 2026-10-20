# Contributing to TADBench

Thanks for your interest in improving TADBench. Bug reports, new detectors,
datasets, and documentation fixes are all welcome.

## Development Setup

Python 3.10-3.12 is required; 3.11 is recommended.

```bash
git clone https://github.com/dortort/TADBench
cd TADBench
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

The test suite uses a deterministic stub embedder, so it needs no model
downloads or prepared datasets. Add or update tests for any behavior change.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
`type(scope): description`, for example `fix(cli): honor --embedder override in matrix`.

Common types are `feat`, `fix`, `docs`, `test`, `build`, `ci`, and `chore`.
Scopes usually name the area touched, such as `cli`, `data`, `detectors`,
`eval`, `metrics`, `runner`, or `deps`.

Keep each commit to a single logical change.

## Pull Requests

- Branch from `main` and open the pull request against `main`.
- `main` is protected: changes land only through pull requests, history stays
  linear (squash or rebase merges), review conversations must be resolved, and
  CodeQL must not report error-level or high/critical-severity alerts.
- Update `README.md` when you change CLI commands, options, or configuration.
- Report security issues privately as described in [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
