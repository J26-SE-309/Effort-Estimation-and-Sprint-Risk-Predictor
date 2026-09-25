import pytest

from erp.tawos import parse_sprint_ids


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("103", [103]),
        ("103, 106", [103, 106]),  # cumulative list after the issue moved to a second sprint
        ("103,106,  110", [103, 106, 110]),
        ("", []),
        (None, []),
        (float("nan"), []),
    ],
)
def test_parse_sprint_ids(value, expected):
    assert parse_sprint_ids(value) == expected
