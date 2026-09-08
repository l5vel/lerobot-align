# Preparing public source

Never push the original development repository's history: removed third-party
annotations remain in old objects. Export a reviewed commit instead:

```bash
uv run --no-sync python scripts/export_release.py /tmp/lerobot-align-public-source
```

The destination must be new. The exporter reads exact committed regular files
from [source-files.txt](source-files.txt), writes a SHA-256 receipt, and copies no
Git objects. Dirty files, local notes, result archives and unlisted tracked files
cannot enter the export. Review any manifest additions as publication changes;
the allowlist is not a credential scanner for the contents of an approved file.

Run the canonical README quick start in the exported directory. Build its wheel
and sdist, then run `bash scripts/check_distributions.sh`. Scan the approved
source and built artifacts before publication. When creating a new public Git
repository, initialize Git **inside this exported directory**, make one new root
commit, and verify `git rev-list --count --all` prints `1`. Do not add a remote
pointing back to the development repository or fetch its history.

The export intentionally omits historical results, analysis documents and
engineering handoff notes. The inventory remains as a hash-only removal record.
No source data is made distributable by being listed in an old Git commit.
