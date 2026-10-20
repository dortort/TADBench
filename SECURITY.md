# Security Policy

## Supported Versions

TADBench has no tagged releases yet. Security fixes are applied to the latest
commit on `main`.

## Reporting a Vulnerability

Please do not open a public issue for security problems. Report them privately
through GitHub's private vulnerability reporting:

https://github.com/dortort/TADBench/security/advisories/new

Include the affected commit, steps to reproduce, and the impact you observed.
This is a single-maintainer project, so reports are handled on a best-effort
basis.

## Scope

In scope: vulnerabilities in the `tadbench` package and CLI, such as arbitrary
code execution, unsafe loading of index, cache, or data files, or a vulnerable
pinned dependency.

Out of scope: prompts that evade the benchmarked detectors. Detection misses
are research results rather than vulnerabilities; please open a regular issue
or pull request with the samples instead.
