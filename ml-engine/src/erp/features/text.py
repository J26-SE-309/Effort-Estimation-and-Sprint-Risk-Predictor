"""Text clues from the story as it was written at commitment.

Three of them stand in for teammates' components until their batch scores exist (ML guide 6.6, route 2):
ambiguity and missing information for Component 1 (Requirement Quality), acceptance criteria and INVEST
checks for Component 2 (Story Refinement). The vague-language indicators follow the categories of the
rule-based baseline in the Component 1 proposal (vague adjectives and adverbs, indefinite quantities,
non-specific time expressions, optional and modal wording) and the unbounded terms listed in
ISO/IEC/IEEE 29148 (loopholes, open-ended and comparative phrases). They are deliberately simple: the
thesis reports them as proxies, and Component 1's real scores replace them when they are available.
"""

import re

import pandas as pd

VAGUE_TERMS = {
    "vague adjective or adverb": [
        "fast", "quick", "quickly", "slow", "easy", "easily", "simple", "simply", "user-friendly", "user friendly",
        "intuitive", "flexible", "robust", "efficient", "efficiently", "effective", "appropriate", "appropriately",
        "adequate", "reasonable", "reasonably", "significant", "significantly", "sufficient", "minimal",
        "optimal", "seamless", "seamlessly", "properly", "correctly", "nice", "good", "clean", "better", "best",
        "improved", "enhanced", "normal", "normally", "typical", "typically",
    ],
    "indefinite quantity": [
        "some", "several", "various", "many", "few", "a lot", "lots of", "most", "a number of", "multiple",
        "numerous", "etc", "and so on", "and/or", "all kinds of",
    ],
    "non-specific time": [
        "soon", "later", "eventually", "sometimes", "often", "usually", "occasionally", "frequently",
        "in the future", "at some point", "as soon as possible", "asap", "in time", "periodically",
    ],
    "optional or modal wording": [
        "if possible", "if needed", "if necessary", "as needed", "as appropriate", "as applicable",
        "where possible", "when possible", "possibly", "maybe", "perhaps", "might", "could",
        "optionally", "ideally", "hopefully",
    ],
    "open-ended or unbounded": [
        "but not limited to", "including but not limited", "as a minimum", "support for",
        "handle", "tbd", "tbc", "to be decided", "to be determined", "and more", "or similar", "something like",
    ],
}
_TERMS = sorted({term for terms in VAGUE_TERMS.values() for term in terms}, key=len, reverse=True)
_VAGUE = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(t) for t in _TERMS) + r")(?![\w-])", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

_AC_HEADING = re.compile(r"acceptance\s+(?:criteria|criterion|tests?)|definition\s+of\s+done", re.I)
_GIVEN_WHEN_THEN = re.compile(r"\bgiven\b.{0,400}?\bwhen\b.{0,400}?\bthen\b", re.I | re.S)
# List items in Jira markup (* item, # item) also survive in TAWOS text whose line breaks were flattened.
_LIST_ITEM = re.compile(r"(?:^|\s)(?:\*+|#+)\s+\S|^\s*(?:-|\d+[.)]|\[[ xX]?\])\s+\S", re.M)
_PLACEHOLDER = re.compile(r"\b(?:tbd|tbc|todo|to be (?:decided|determined|defined))\b|\?\?\?", re.I)
_USER_STORY = re.compile(r"\bas an?\b.{1,120}?\bi (?:want|need|would like)\b", re.I | re.S)
_SO_THAT = re.compile(r"\bso that\b", re.I)
_STEPS = re.compile(r"steps? to reproduce|expected (?:result|behaviou?r)|actual (?:result|behaviou?r)|"
                    r"to reproduce|repro(?:duction)? steps", re.I)
_TESTS = re.compile(r"\b(?:unit|integration|regression|acceptance|end-to-end|e2e|functional)\s+tests?\b|"
                    r"\btest\s+(?:case|plan|coverage|suite)s?\b", re.I)

SHORT_DESCRIPTION_WORDS = 10


def vague_terms(text: str) -> list[str]:
    return [match.group(0).lower() for match in _VAGUE.finditer(text or "")]


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text or "") if _WORD.search(s)]


def ambiguity_score(text: str) -> float:
    """Share of sentences with at least one vague term (0 = none, 1 = every sentence)."""
    parts = sentences(text)
    return sum(1 for s in parts if _VAGUE.search(s)) / len(parts) if parts else 0.0


def acceptance_criteria(description: str) -> tuple[bool, int]:
    """Whether the description has acceptance criteria, and how many criteria it lists.

    Criteria count as present with an 'Acceptance criteria' / 'Definition of done' heading or a
    Given / When / Then structure. Their number is the list items after the heading (or the number of
    'Given' clauses), at least 1.
    """
    text = description or ""
    heading = _AC_HEADING.search(text)
    gwt = _GIVEN_WHEN_THEN.search(text)
    if not heading and not gwt:
        return False, 0
    items = len(_LIST_ITEM.findall(text[heading.end():])) if heading else 0
    givens = len(re.findall(r"\bgiven\b", text, re.I)) if gwt else 0
    return True, max(1, items, givens)


def missing_info_flags(title: str, description: str, issue_type: str) -> int:
    """Count of simple 'something is missing' signs (Component 1 proxy).

    1. the description is empty or under SHORT_DESCRIPTION_WORDS words;
    2. it still holds placeholders (TBD, TODO, ???);
    3. a bug without steps to reproduce or expected / actual behaviour;
    4. a story without a user or a goal (no 'As a ... I want' and no 'so that').
    """
    text = description or ""
    flags = int(len(_WORD.findall(text)) < SHORT_DESCRIPTION_WORDS)
    flags += int(bool(_PLACEHOLDER.search(f"{title} {text}")))
    if issue_type == "Bug":
        flags += int(not _STEPS.search(text))
    if issue_type == "Story":
        flags += int(not (_USER_STORY.search(f"{title} {text}") or _SO_THAT.search(text)))
    return flags


def text_features(stories: pd.DataFrame) -> pd.DataFrame:
    """Text clues per story. stories needs title, description (raw, with line breaks), description_text
    (code removed) and issue_type, as in the snapshot."""
    rows = []
    for title, raw, clean, kind in zip(stories["title"], stories["description"].fillna(""),
                                       stories["description_text"], stories["issue_type"], strict=True):
        has_ac, n_criteria = acceptance_criteria(raw)
        full = f"{title}. {clean}"
        rows.append({
            "title_length": len(_WORD.findall(title)),
            "description_length": len(_WORD.findall(clean)),
            "description_has_code": "{code" in raw.lower() or "{noformat" in raw.lower(),
            "vague_term_count": len(vague_terms(full)),
            "ambiguity_score": round(ambiguity_score(full), 4),
            "missing_info_flag_count": missing_info_flags(title, clean, kind),
            "has_acceptance_criteria": has_ac,
            "ac_criteria_count": n_criteria,
            "ac_completeness_score": round(min(1.0, n_criteria / 3), 4),
            "user_story_format": bool(_USER_STORY.search(f"{title} {clean}")),
            "states_goal": bool(_SO_THAT.search(clean)),
            "mentions_tests": bool(_TESTS.search(clean)),
        })
    return pd.DataFrame(rows, index=stories.index)
