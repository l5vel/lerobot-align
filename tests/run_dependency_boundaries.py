"""Run the real-media CLI smoke with dependency-advisory tripwires installed."""

import json
from pathlib import Path
import sys
import tempfile


def main() -> int:
    from tests.dependency_boundaries import rejected_advisory_operations, watched_operations

    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Import trusted installed application code before accepting dataset
        # input. This wrapper invokes the same function as the console script.
        from importlib.metadata import entry_points

        entry = next(p for p in entry_points(group="console_scripts") if p.name == "lerobot-align")
        cli = entry.load()
        evidence = Path(sys.argv[2])
        sys.argv = ["lerobot-align", *sys.argv[3:]]
        with rejected_advisory_operations() as attempts:
            cli()
            assert not attempts, attempts
        with evidence.open("a") as stream:
            stream.write(json.dumps({"guarded_operations": len(watched_operations()), "attempts": attempts}) + "\n")
        return 0

    from tests.run_e2e_smoke import main as smoke

    with tempfile.TemporaryDirectory(prefix="align-dependency-boundaries-") as directory:
        evidence = Path(directory) / "guarded-cli.jsonl"
        smoke(cli_prefix=[sys.executable, "-m", "tests.run_dependency_boundaries", "--cli", str(evidence)])
        records = [json.loads(line) for line in evidence.read_text().splitlines()]
        assert len(records) == 2, records
        assert all(row["guarded_operations"] == 14 and not row["attempts"] for row in records)
        print("Dependency boundaries: both visual CLI workflows passed with all 14 tripwires active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
