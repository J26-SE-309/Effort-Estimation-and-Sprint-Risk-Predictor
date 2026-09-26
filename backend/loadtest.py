"""NFR1 load test: a 50-story backlog within 2 s at the 95th percentile under 10 concurrent users.

Run against a running service (docker compose up, or uvicorn):
    python loadtest.py                                  # http://127.0.0.1:8004, 10 users x 10 requests
    python loadtest.py --url http://host:8004 --users 10 --requests 20 --stories 50
Every user sends its backlog again as soon as the previous answer arrives; the report gives the response-time
percentiles over all requests (after one warm-up request per user).

Every prediction is written to the service's audit log (about 2.3 KB a story; the default run adds 5,500 rows), so
the test refuses to run against a service using the hosted database (Neon's free plan holds 0.5 GB) unless told
to: run the service on the local database, e.g. with DATABASE_URL set to the local effort-db.
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BUDGET_SECONDS = 2.0


def backlog(stories: int, user: int) -> dict:
    return {
        "project_id": f"LOAD-{user}",
        "stories": [{"story_id": f"U{user}-S{i}", "title": f"Story {i}: export report type {i % 7} for user {user}",
                     "description": "As an analyst I want to export the report so that I can share it. It must be "
                                    "fast and handle large reports.",
                     "story_points": [1, 2, 3, 5, 8][i % 5], "blocker_count": int(i % 6 == 0)}
                    for i in range(stories)],
        "team_context": {"velocity_mean": 30, "velocity_variance": 25, "closed_sprints": 10},
    }


def post(url: str, payload: dict) -> float:
    request = urllib.request.Request(f"{url}/api/v1/estimate", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
    return time.perf_counter() - started


def database_location(url: str) -> str:
    with urllib.request.urlopen(f"{url}/health", timeout=30) as response:
        return json.loads(response.read()).get("database_location", "local")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8004")
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--stories", type=int, default=50)
    parser.add_argument("--allow-hosted-database", action="store_true",
                        help="run even though the service records into the hosted database")
    args = parser.parse_args()
    if database_location(args.url) == "hosted" and not args.allow_hosted_database:
        rows = args.users * (args.requests + 1) * args.stories
        raise SystemExit(f"The service records into the hosted database: this run would add {rows:,} audit rows "
                         "(~2.3 KB each). Point the service at the local database, or pass --allow-hosted-database.")

    times: list[float] = []
    lock = threading.Lock()
    start_together = threading.Barrier(args.users)

    def user(number: int) -> None:
        payload = backlog(args.stories, number)
        post(args.url, payload)  # warm-up
        start_together.wait()
        for _ in range(args.requests):
            elapsed = post(args.url, payload)
            with lock:
                times.append(elapsed)

    started = time.perf_counter()
    with ThreadPoolExecutor(args.users) as pool:
        list(pool.map(user, range(args.users)))
    wall = time.perf_counter() - started
    ordered = sorted(times)
    p95 = ordered[max(0, int(round(0.95 * len(ordered))) - 1)]
    print(f"{len(times)} requests of {args.stories} stories from {args.users} concurrent users in {wall:.1f} s")
    print(f"response time: median {statistics.median(ordered):.2f} s, p95 {p95:.2f} s, max {ordered[-1]:.2f} s")
    print(f"NFR1 (p95 under {BUDGET_SECONDS:g} s): {'met' if p95 < BUDGET_SECONDS else 'NOT met'}")


if __name__ == "__main__":
    main()
