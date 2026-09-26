"""Synthetic backlogs for development, tests and demos: estimate requests with made-up story text, upstream
signals (Components 1-3) and tracker links.

Those inputs arrive with every estimate request (only the sprint history is stored, app.history), so the
synthetic data for them is requests, kept as files:
    python -m app.devdata backlogs            # -> backend/examples/backlogs/*.json

- one backlog per SYN-* team (app.devdata), for its running sprint, so the team's stored history is used;
- the steady team's backlog without the components' signals, and with Component 1's only (FR17);
- odd edge cases (ML guide 7.2): an empty description, a 5,000-word story, Sinhala, Tamil and emoji, extreme
  points and signals, an unknown project.

The stories are written for the platform's example domain (a tutoring platform) from templates, by a seeded
random generator: the same files every time. Each story's signals follow its kind: a vague story gets a high
ambiguity score and no acceptance criteria, a clear one the opposite, with some noise. Synthetic: for
development, integration tests, demos and user-study scenarios only, never to evaluate the models (ML guide 7.2);
named in the proposal's AI-use disclosure (Appendix H).
"""

import json
import zlib
from pathlib import Path

import numpy as np

LABEL = ("Synthetic backlog (backend/app/backlogs.py): made-up stories, component signals and links for "
         "development, tests and demos. Never use it to evaluate the models (ML guide 7.2).")
OUT = Path(__file__).resolve().parents[1] / "examples" / "backlogs"
POINTS = [1, 2, 3, 5, 8, 13]
KINDS = ("clear", "partial", "vague", "bug", "task")
PRIORITIES = (["Major", "Minor", "Critical", "Trivial", "Blocker"], [0.5, 0.25, 0.12, 0.08, 0.05])

# role, goal, benefit, acceptance criteria (given, when, then), typical points
FEATURES = [
    ("student", "book an available tutoring session by choosing a tutor, date and time", "I get help when it suits me",
     [("I am logged in and a tutor has a free slot", "I pick the slot and confirm",
       "the session is booked and both of us get an email"),
      ("the slot was taken a moment ago", "I confirm", "I am told it is no longer free and can pick another")], 5),
    ("tutor", "set my weekly availability", "students can only book times I can teach",
     [("I am on my availability page", "I mark Monday 16:00-18:00 as free and save",
       "students see those hours as bookable")], 3),
    ("student", "cancel a booked session up to 24 hours before it starts", "I am not charged for a session I miss",
     [("my session starts in more than 24 hours", "I cancel it", "it is removed and I am not charged"),
      ("my session starts in less than 24 hours", "I try to cancel", "I am told the cancellation window has passed")],
     3),
    ("student", "see my upcoming sessions on a calendar", "I do not miss any",
     [("I have two booked sessions", "I open the calendar", "both appear on their dates with the tutor's name")], 2),
    ("admin", "export the month's sessions as a CSV file", "finance can pay the tutors",
     [("sessions took place this month", "I export them", "the file lists each session's tutor, date and price")], 3),
    ("student", "rate a tutor after a session", "other students can choose good tutors",
     [("my session has ended", "I give it 1 to 5 stars", "the tutor's average rating includes mine")], 2),
    ("tutor", "upload course materials for a session", "students can prepare",
     [("I have a booked session", "I upload a PDF of up to 20 MB", "the student can download it from the booking")],
     5),
    ("parent", "receive a weekly progress summary for my child", "I know how they are doing",
     [("my child had sessions this week", "Sunday evening comes", "I get an email listing the sessions and ratings")],
     8),
    ("student", "pay for a session with a saved card", "checking out is quick",
     [("I saved a card before", "I pay for a booking", "the saved card is charged without re-entering it"),
      ("the card is declined", "I pay", "the booking is held for 15 minutes while I try another card")], 8),
    ("admin", "suspend a tutor's account", "reported tutors cannot take bookings while we investigate",
     [("a tutor is active", "I suspend the account", "their profile is hidden and new bookings are refused")], 3),
    ("student", "search tutors by subject and hourly price", "I find someone I can afford",
     [("tutors teach maths at different prices", "I search maths up to 20 per hour",
       "only maths tutors at or below 20 are listed")], 5),
    ("tutor", "message a student before a session", "we can agree what to cover",
     [("I have a booked session with a student", "I send a message", "the student sees it in their inbox")], 5),
]
VAGUE = [
    ("Improve the booking page", "The booking page should be faster and more user-friendly. Details TBD."),
    ("Better notifications", "Users complain about notifications. Make them better, as appropriate."),
    ("Payments rework", "Rework payments so they are more flexible etc. Talk to finance at some point."),
    ("Fix the calendar", "The calendar is confusing sometimes. It needs to be simple and intuitive."),
    ("Tutor dashboard", "Tutors need a dashboard with the usual stuff."),
    ("Performance", "Everything should load quickly, ASAP."),
    ("Reports for admins", "Some reports, to be decided with the admins."),
    ("Mobile support", "Make the site work nicely on phones where possible."),
]
BUGS = [
    ("Booking confirmation email not sent", "Steps: book a session as a student. Expected: an email within a "
     "minute. Actual: nothing arrives for about 1 in 10 bookings."),
    ("Calendar shows sessions an hour late after the clock change", "Since the daylight-saving change, sessions "
     "appear one hour later in the student calendar. Tutors' calendars are correct."),
    ("Crash when uploading a file larger than 20 MB", "Uploading course materials over 20 MB returns HTTP 500.\n\n"
     "```\nTypeError: cannot read 'size' of undefined\n    at upload.js:88\n```"),
    ("Double charge when Pay is clicked twice", "Clicking Pay twice quickly charges the card twice."),
    ("Tutor search ignores the price filter", "Filtering tutors by a maximum price still lists tutors above it."),
]
TASKS = [
    ("Upgrade the database driver", "The PostgreSQL driver is two major versions behind. Upgrade it and run the "
     "full test suite."),
    ("Add monitoring for failed payments", "Alert the on-call developer when more than 5 payments fail in 10 "
     "minutes."),
    ("Document the booking API", "Write request and response examples for every /bookings endpoint."),
    ("Set up nightly database backups", "Back up the database every night and keep 14 days of backups."),
]

# stories in the backlog, the share of each kind (KINDS order), the chance a story is blocked
TEAMS = {
    "SYN-NEW": (8, [0.4, 0.3, 0.1, 0.1, 0.1], 0.1),
    "SYN-ONE": (10, [0.4, 0.3, 0.1, 0.1, 0.1], 0.1),
    "SYN-TWO": (10, [0.35, 0.3, 0.15, 0.1, 0.1], 0.15),
    "SYN-STEADY": (12, [0.5, 0.25, 0.05, 0.1, 0.1], 0.05),
    "SYN-ERRATIC": (15, [0.15, 0.25, 0.35, 0.15, 0.1], 0.35),
    "SYN-NOPOINTS": (8, [0.3, 0.3, 0.2, 0.1, 0.1], 0.15),
}


def _signals(rng: np.random.Generator, kind: str, story: dict) -> dict:
    """What Components 1-3 would say about the story, consistent with its kind (plus noise)."""
    def between(low: float, high: float) -> float:
        return round(float(rng.uniform(low, high)), 2)

    has_ac = bool(story["acceptance_criteria"])
    quality = {
        "clear": (between(0.05, 0.2), 0, 0, between(0.8, 1.0)),
        "partial": (between(0.3, 0.5), int(rng.integers(0, 2)), 1, between(0.3, 0.6) if has_ac else 0.0),
        "vague": (between(0.7, 0.95), int(rng.integers(2, 5)), int(rng.integers(2, 4)), 0.0),
        "bug": (between(0.15, 0.35), 0, int(rng.integers(0, 2)), 0.0),
        "task": (between(0.1, 0.3), 0, 0, 0.0),
    }[kind]
    tests = bool(rng.random() < {"clear": 0.6, "bug": 0.5}.get(kind, 0.2))
    traces = [bool(story["has_epic"]), (story["linked_issue_count"] or 0) > 0, tests]
    points = story["story_points"]
    return {
        "ambiguity_score": quality[0], "vague_term_count": quality[1], "missing_info_flag_count": quality[2],
        "ac_completeness_score": quality[3],
        "invest_compliance_flags": {
            "independent": story["blocker_count"] == 0, "negotiable": kind != "vague",
            "valuable": kind in ("clear", "partial"), "estimable": kind != "vague",
            "small": points is not None and points <= 8, "testable": has_ac or kind == "bug"},
        "traceability_coverage_pct": round(sum(traces) / 3, 2), "unlinked_artifact_count": 3 - sum(traces),
        "has_linked_tests": tests,
    }


def _pick(rng: np.random.Generator, pool: list, used: set[int]):
    """A template not used yet in this backlog (all of them again once each has been used)."""
    free = [i for i in range(len(pool)) if i not in used] or list(range(len(pool)))
    index = free[int(rng.integers(len(free)))]
    used.add(index)
    return pool[index]


def _story(rng: np.random.Generator, story_id: str, kind: str, blocked: float, estimated: bool,
           used: dict[str, set[int]]) -> dict:
    if kind in ("clear", "partial"):
        role, goal, benefit, criteria, typical = _pick(rng, FEATURES, used.setdefault("feature", set()))
        article = "an" if role[0] in "aeiou" else "a"
        title = f"As {article} {role} I want to {goal} so that {benefit}"
        acs = [f"Given {g}, when {w}, then {t}" for g, w, t in criteria]
        if kind == "partial":
            acs = acs[:1] if rng.random() < 0.5 else []
        description = (f"{role.capitalize()}s have asked for this in the last survey." if kind == "clear"
                       else "See the survey.")
        points, issue_type = typical, "Story"
    else:
        pool = {"vague": VAGUE, "bug": BUGS, "task": TASKS}[kind]
        title, description = _pick(rng, pool, used.setdefault(kind, set()))
        acs = []
        points = {"vague": int(rng.choice([8, 13])), "bug": int(rng.choice([1, 2, 3])),
                  "task": int(rng.choice([2, 3, 5]))}[kind]
        issue_type = {"vague": "Improvement", "bug": "Bug", "task": "Task"}[kind]
    blockers = int(rng.choice([1, 2])) if rng.random() < blocked else 0
    dep_out, dep_in = int(rng.integers(0, 3)), int(rng.integers(0, 3))
    story = {
        "story_id": story_id, "title": title, "description": description, "acceptance_criteria": acs,
        "issue_type": issue_type, "priority": str(rng.choice(PRIORITIES[0], p=PRIORITIES[1])),
        "story_points": float(points) if estimated and not (kind == "vague" and rng.random() < 0.5) else None,
        "blocker_count": blockers, "dep_in_degree": dep_in, "dep_out_degree": max(dep_out, blockers),
        "linked_issue_count": max(dep_out, blockers) + dep_in + int(rng.integers(0, 2)),
        "has_epic": bool(rng.random() < (0.4 if kind == "vague" else 0.85)),
        "in_progress": bool(rng.random() < 0.1), "added_mid_sprint": False, "days_into_sprint": 0.0,
    }
    story["upstream"] = _signals(rng, kind, story)
    return story


def team_backlog(name: str) -> dict:
    """The team's backlog for its running sprint (app.devdata: SYN-X-S<closed + 1>)."""
    from app.devdata import SYNTHETIC

    size, shares, blocked = TEAMS[name]
    rng = np.random.default_rng(zlib.crc32(f"backlog {name}".encode()))
    kinds = rng.choice(KINDS, size=size, p=shares)
    used: dict[str, set[int]] = {}
    stories = [_story(rng, f"{name}-B{i + 1}", str(kind), blocked, name != "SYN-NOPOINTS", used)
               for i, kind in enumerate(kinds)]
    return {"_synthetic": LABEL, "project_id": name, "sprint_id": f"{name}-S{SYNTHETIC[name].closed + 1}",
            "stories": stories}


def without_components(backlog: dict, keep: tuple[str, ...] = ()) -> dict:
    """The same backlog with the components' signals left out (FR17), except the fields in `keep`."""
    stories = [{**s, "upstream": {k: v for k, v in s["upstream"].items() if k in keep}} for s in backlog["stories"]]
    return {**backlog, "stories": stories}


def edge_cases() -> dict:
    """Odd stories the service must still answer (ML guide 7.2), for a project it has never seen."""
    base = {"acceptance_criteria": [], "issue_type": "Story", "priority": "Major", "story_points": 3.0}
    sentence = ("The tutor can share a whiteboard, and the student can draw on it while they talk about the "
                "exercise. ")
    long_text = (sentence * 400).strip()  # 5,200 words
    stories = [
        {**base, "story_id": "EDGE-1", "title": "Set up the project repository", "description": ""},
        {**base, "story_id": "EDGE-2", "title": "As a tutor I want a shared whiteboard so that we can work together",
         "description": long_text},
        {**base, "story_id": "EDGE-3", "title": "As a student I want lesson titles in Sinhala (පාඩම්) and Tamil "
         "(பாடம்) 📚 so that I can read them in my language", "description": "සිංහල සහ தமிழ் 🙂"},
        {**base, "story_id": "EDGE-4", "title": "Refactor the booking service",
         "description": "```python\ndef book(slot):\n    return Booking(slot)\n```\n" * 20},
        {**base, "story_id": "EDGE-5", "title": "Fix", "description": "   \n\t  ", "story_points": 0.0},
        {**base, "story_id": "EDGE-6", "title": "Rebuild the whole platform", "story_points": 100.0,
         "description": "Everything, end to end."},
        {**base, "story_id": "EDGE-7", "title": "As a student I want to reset my password so that I can log in",
         "story_points": None, "acceptance_criteria": ["Given I forgot my password, when I ask for a reset, then I "
                                                       "get an email with a link valid for one hour"]},
        {**base, "story_id": "EDGE-8", "title": "Integrate the payment provider", "description": "Blocked a lot.",
         "blocker_count": 5, "dep_out_degree": 10, "dep_in_degree": 7, "linked_issue_count": 25},
        {**base, "story_id": "EDGE-9", "title": "Make it better", "description": "TBD",
         "upstream": {"ambiguity_score": 1.0, "vague_term_count": 12, "missing_info_flag_count": 9,
                      "ac_completeness_score": 0.0, "traceability_coverage_pct": 0.0, "unlinked_artifact_count": 3,
                      "has_linked_tests": False}},
        {**base, "story_id": "EDGE-10", "title": "As an admin I want audit logs so that I can see who changed what",
         "upstream": {"invest_compliance_flags": {"is_independent": True, "Testable": False, "VALUABLE": True}}},
        {**base, "story_id": "EDGE-11", "title": "Add dark mode", "issue_type": "Epic", "priority": "Unknown"},
    ]
    return {"_synthetic": LABEL, "project_id": "SYN-EDGE", "stories": stories}


def all_backlogs() -> dict[str, dict]:
    """File name -> request."""
    files = {f"{name.lower()}.json": team_backlog(name) for name in TEAMS}
    steady = files["syn-steady.json"]
    files["syn-steady-without-components.json"] = without_components(steady)
    files["syn-steady-component-1-only.json"] = without_components(
        steady, keep=("ambiguity_score", "vague_term_count", "missing_info_flag_count"))
    files["edge-cases.json"] = edge_cases()
    return files


def write(out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, backlog in all_backlogs().items():
        path = out / name
        path.write_text(json.dumps(backlog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    return written
