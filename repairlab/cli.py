"""Dependency-free host CLI. Candidate execution always requires Docker."""

import argparse
import json
from pathlib import Path

from .execution import DockerExecutor, InfrastructureError
from .experiments import admission, demo, score_patches, write_json
from .tasks import manifest, pilot_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the development tasks and checked domains")
    export = commands.add_parser("export-public", help="Export only agent-visible task material")
    export.add_argument("--output", required=True, type=Path)
    admit = commands.add_parser("admit", help="Validate task controls; these are not model results")
    admit.add_argument("--output", required=True, type=Path)
    replay = commands.add_parser("demo", help="Save a clearly labeled scripted repair replay")
    replay.add_argument("--task", default="quantity-range")
    replay.add_argument("--output", required=True, type=Path)
    score = commands.add_parser("score-patches", help="Grade a single-response model baseline")
    score.add_argument("--predictions", required=True, type=Path)
    score.add_argument("--metadata", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    external_admit = commands.add_parser("external-admit", help="Check HumanEvalFix controls")
    external_admit.add_argument("--output", required=True, type=Path)
    external_score = commands.add_parser("external-score", help="Score the HumanEvalFix adaptation")
    external_score.add_argument("--predictions", required=True, type=Path)
    external_score.add_argument("--metadata", required=True, type=Path)
    external_score.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps(manifest(), indent=2, ensure_ascii=False))
        return
    if args.command == "export-public":
        if args.output.exists():
            parser.error("Choose a new output directory")
        args.output.mkdir(parents=True)
        write_json(args.output / "public-tasks.json", [t.public() for t in pilot_tasks()])
        write_json(args.output / "manifest.json", manifest())
        return
    if args.output.exists():
        parser.error("Choose a new output directory; existing experiment evidence is preserved")
    try:
        executor = DockerExecutor()
        if args.command == "admit":
            result = admission(executor, args.output)
            if not result["all_matched"] or not result["source_unchanged"]:
                raise ValueError("Admission controls failed or source changed during execution")
        elif args.command == "demo":
            tasks = {t.id: t for t in pilot_tasks()}
            if args.task not in tasks:
                parser.error("Unknown task")
            result = demo(executor, tasks[args.task], args.output)
        elif args.command in {"external-admit", "external-score"}:
            from . import humanevalfix

            if args.command == "external-admit":
                result = humanevalfix.admit(executor, args.output)
            else:
                result = humanevalfix.score(executor, args.predictions, args.metadata, args.output)
        else:
            result = score_patches(executor, args.predictions, args.metadata, args.output)
        if result.get("source_unchanged") is False:
            raise ValueError("Source changed during execution; results are not a frozen run")
        write_json(args.output / "status.json", {"status": "completed"})
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "successes": result.get("successes"),
                    "all_matched": result.get("all_matched"),
                },
                indent=2,
            )
        )
    except (InfrastructureError, ValueError, OSError) as exc:
        if args.output.exists() and (args.output / "status.json").exists():
            write_json(
                args.output / "status.json",
                {
                    "status": "infrastructure_error"
                    if isinstance(exc, InfrastructureError)
                    else "error",
                    "error": str(exc),
                },
            )
        parser.exit(2, f"RepairLab stopped: {exc}\n")


if __name__ == "__main__":
    main()
