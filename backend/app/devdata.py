"""Development data for the sprint history, until the platform serves its sprints: real TAWOS sprints and a small
synthetic set, loaded through the same CSV import as real records, and removed again by source.

Run from backend/ (into the service's database: DATABASE_URL in backend/.env, else the local effort-db):
    python -m app.devdata load                   # the TAWOS projects in TAWOS_PROJECTS and the SYN-* teams
    python -m app.devdata load --tawos MESOS XD  # these TAWOS projects (as TAWOS-MESOS, TAWOS-XD) only
    python -m app.devdata load --synthetic       # the SYN-* teams only
    python -m app.devdata list
    python -m app.devdata delete                 # every tawos and synthetic record, and those projects'
                                                 # predictions, feedback, outcomes and pins
    python -m app.devdata delete --source synthetic
    python -m app.devdata backlogs               # the synthetic backlogs (estimate requests) as files, app.backlogs

The TAWOS projects are real history from the TAWOS dataset (they need the Datasets folder). The SYN-* teams are
made up by a seeded random generator for tests and demos: labelled synthetic, never used in any evaluation (ML
guide 7.2), and to be named in the proposal's AI-use disclosure (Appendix H).
"""

import argparse
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import delete, select

from app import db, history, store
from app.tables import FeedbackRecord, OutcomeRecord, PinnedConfiguration, PredictionRecord

TAWOS_PROJECTS = ("MESOS", "INDY")
DEVELOPMENT = ("tawos", "synthetic")
POINTS = np.array([1, 2, 3, 5, 8])
DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Team:
    closed: int              # sprints already closed
    velocity: float | None   # typical points committed a sprint (None: the team never estimates)
    spread: float            # how much that varies, as a share
    spill: float             # chance a story is not done by the end of its sprint
    reopen: float            # chance a done story is reopened


SYNTHETIC = {
    "SYN-NEW": Team(0, 20, 0.15, 0.2, 0.05),       # a brand-new team: only its first sprint, running
    "SYN-ONE": Team(1, 20, 0.15, 0.2, 0.05),       # one sprint closed
    "SYN-TWO": Team(2, 24, 0.15, 0.2, 0.05),       # two: one short of leaving the cold start
    "SYN-STEADY": Team(8, 30, 0.10, 0.12, 0.03),   # predictable
    "SYN-ERRATIC": Team(8, 26, 0.50, 0.40, 0.12),  # unpredictable
    "SYN-NOPOINTS": Team(4, None, 0.15, 0.25, 0.05),
}


def synthetic(name: str, team: Team, today: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A made-up team: `team.closed` two-week sprints back to back, then one running for 3 days by `today`.
    Stories not done by the end are carried into the next sprint. Seeded by the name: the same team every time."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    running_start = today.normalize() - 3 * DAY + pd.Timedelta(hours=9)
    sprints, items, carried = [], [], []
    number = 0
    for k in range(team.closed + 1):
        start = running_start - 14 * DAY * (team.closed - k)
        end, running = start + 14 * DAY, k == team.closed
        sprints.append({"sprint_id": f"{name}-S{k + 1}", "name": f"Sprint {k + 1}", "started_at": start,
                        "planned_end": end, "closed_at": None if running else end + pd.Timedelta(hours=7)})
        target = max(5.0, (team.velocity or 20) * (1 + team.spread * rng.standard_normal()))
        planned = []
        while sum(p for _, p in planned) < target:
            number += 1
            planned.append((f"{name}-{number}", float(rng.choice(POINTS, p=[0.15, 0.25, 0.3, 0.2, 0.1]))))
        stories = [(story, points, False) for story, points in carried] + [(s, p, True) for s, p in planned]
        carried = []
        for story, points, first in stories:
            mid_sprint = first and not running and rng.random() < 0.15
            committed = (start + (rng.uniform(1, 9) * DAY if mid_sprint else pd.Timedelta(0))).floor("min")
            started = (committed + rng.uniform(0, 2) * DAY).floor("min")
            now_or_end = min(today, end) if running else end
            done = (rng.random() > team.spill) if not running else (started < now_or_end and rng.random() < 0.3)
            resolved = (started + rng.uniform(0.2, 0.9) * (now_or_end - started)).floor("min") if done else None
            if started > now_or_end:
                started = None
            row = {"sprint_id": f"{name}-S{k + 1}", "story_id": story,
                   "issue_type": rng.choice(["Story", "Task", "Bug"], p=[0.6, 0.25, 0.15]),
                   "committed_at": committed, "points_at_commit": points if team.velocity else None,
                   "points_at_close": None if running or not team.velocity else points,
                   "done_in_sprint": None if running else bool(done),
                   "spilled_over": None if running or not first else not done,
                   "reopened": None if running or not first else bool(done and rng.random() < team.reopen),
                   "started_at": started, "resolved_at": resolved,
                   "hours_in_progress": round((resolved - started) / pd.Timedelta(hours=1) * 0.6, 1) if done
                   else None}
            items.append(row)
            if not done and not running:
                carried.append((story, points))
    return pd.DataFrame(sprints), pd.DataFrame(items)


def load(sprints: pd.DataFrame, items: pd.DataFrame, project_id: str, source: str) -> dict:
    """Through the CSV import, like any records: the same checks, the same format."""
    parsed_sprints, parsed_items, problems = history.parse_csv(history.to_csv(sprints, items))
    if problems:
        raise SystemExit(f"{project_id}: the records do not pass the import's checks: {problems[:5]}")
    return history.replace(project_id, parsed_sprints, parsed_items, source)


def load_tawos(keys: list[str]) -> None:
    from erp.serving import replay

    sprints, items, _ = replay.records(keys, prefix="TAWOS-")
    for project, own in items.groupby("project_id", sort=False):
        counts = load(sprints[sprints["project_id"] == project], own, project, "tawos")
        print(f"{project}: {counts['sprints']} sprints, {counts['stories']:,} stories ({counts['rows']:,} rows)")


def load_synthetic(today: pd.Timestamp | None = None) -> None:
    today = today or pd.Timestamp(datetime.now(UTC))
    for name, team in SYNTHETIC.items():
        counts = load(*synthetic(name, team, today), name, "synthetic")
        print(f"{name}: {counts['sprints']} sprints, {counts['stories']:,} stories")


def remove(sources: tuple[str, ...]) -> list[str]:
    """Development records, and the predictions made for those projects (their audit log, feedback, outcomes
    and pins)."""
    projects: list[str] = []
    for source in sources:
        projects += history.remove(source=source)
    if projects:
        with db.engine.begin() as connection:
            ids = select(PredictionRecord.id).where(PredictionRecord.project_id.in_(projects))
            connection.execute(delete(FeedbackRecord).where(FeedbackRecord.prediction_id.in_(ids)))
            connection.execute(delete(OutcomeRecord).where(OutcomeRecord.prediction_id.in_(ids)))
            connection.execute(delete(PredictionRecord).where(PredictionRecord.project_id.in_(projects)))
            connection.execute(delete(PinnedConfiguration).where(PinnedConfiguration.project_id.in_(projects)))
    return sorted(set(projects))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    loading = commands.add_parser("load", help="load development data (both kinds unless one is named)")
    loading.add_argument("--tawos", nargs="*", metavar="KEY", help=f"TAWOS projects (default {TAWOS_PROJECTS})")
    loading.add_argument("--synthetic", action="store_true", help="the SYN-* teams")
    commands.add_parser("list", help="the projects with sprint records and their sources")
    removing = commands.add_parser("delete", help="remove development data and those projects' predictions")
    removing.add_argument("--source", choices=DEVELOPMENT, help="only this kind (default: both)")
    commands.add_parser("backlogs", help="write the synthetic backlogs to backend/examples/backlogs")
    args = parser.parse_args(argv)
    if args.command == "backlogs":  # files only: no database
        from app import backlogs

        for path in backlogs.write():
            print(f"Wrote {path}")
        return

    where = "hosted" if db.hosted else "local"
    if not store.migrate():  # the tables as the service makes them at start-up
        raise SystemExit(f"The {where} database cannot be reached or migrated.")
    if args.command == "load":
        both = args.tawos is None and not args.synthetic
        if args.tawos is not None or both:
            load_tawos(args.tawos or list(TAWOS_PROJECTS))
        if args.synthetic or both:
            load_synthetic()
        print(f"Loaded into the {where} database.")
    elif args.command == "list":
        for project, sources in history.projects().items():
            print(f"{project}: {', '.join(sources)}")
    else:
        gone = remove((args.source,) if args.source else DEVELOPMENT)
        print(f"Removed from the {where} database: {', '.join(gone) or 'nothing'}")


if __name__ == "__main__":
    main()
