# Publishing a dataset release

GitHub is the canonical home for implementation and development. Hugging Face
hosts versioned prepared data. The exporter does not upload anything and does not
run expensive model inference.

## Build

Use the project's Python 3.10–3.12 environment with its dependencies installed:

```bash
python -m tadbench.data.release \
  --output-dir data/releases/v0.1.0 \
  --version v0.1.0 \
  --repo-id dortort/TADBench
```

The destination must not exist. This prevents stale splits and provenance from
being accidentally reused. Each build freshly downloads the exact upstream
revisions in `tadbench/config.py`, transforms/generates/splits inputs, audits
schemas and held-out disjointness, and creates:

```text
data/releases/v0.1.0/
  raw/          # Local upstream Parquet downloads
  processed/    # Local preparation outputs
  dataset/      # The only directory to upload
    README.md
    NOTICE.md
    TADBench-LICENSE.txt
    licenses/
    manifest.json
    provenance/
      source/   # Preparation code snapshot, also fingerprinted in the manifest
    data/
```

Commit code changes before producing the final export so the manifest records
a clean, available source revision. UUIDs vary across builds; preserve the
archived JSONL for byte-identical reproduction. Do not upload the project root,
local raw folder, credentials, caches, or model indexes.

If a build fails partway, for example on a network error, its output directory
is left behind and blocks reruns by design. Delete it and rerun the full build:
partial outputs are never reused.

## Verify and publish

Review the generated card, attribution, counts, and manifest. Load every named
configuration locally and verify checksums with:

```bash
python -m tadbench.data.release --verify data/releases/v0.1.0/dataset
```

The build command runs this verification automatically before reporting success.
Keep the October 2026 result tables labeled historical; this data release does
not establish corrected scores.

The `code_revision` recorded in `manifest.json` must stay reachable after the
release PR merges. `main` only allows squash or rebase merges, which rewrite
commit SHAs, and merged branches are deleted. Before merging, tag that revision
and push the tag:

```bash
git tag -a dataset/v0.1.0 "$(jq -r .code_revision data/releases/v0.1.0/dataset/manifest.json)" \
  -m "Code revision for the TADBench v0.1.0 dataset release"
git push origin dataset/v0.1.0
```

Install the official CLI once and authenticate in the browser:

```bash
brew install hf
hf auth login
hf auth whoami
```

Create the repository, upload only the export, and tag the verified commit:

```bash
hf repos create dortort/TADBench --repo-type dataset --public
hf upload dortort/TADBench data/releases/v0.1.0/dataset . \
  --repo-type dataset \
  --commit-message 'feat(data): publish TADBench v0.1.0'
# Verify remote downloads at the returned upload commit before tagging.
hf repos tag create dortort/TADBench v0.1.0 \
  --repo-type dataset --revision UPLOAD_COMMIT_SHA
```

If the repository already exists, inspect it before updating. Never overwrite
an existing release tag. Test loading all configurations at the uploaded commit
and download/hash-check the archived JSONL. Record the resolved dataset commit
in future experiment reports. A later GitHub results release should include
predictions, run manifests, hardware details, package versions, and that same
dataset commit.
