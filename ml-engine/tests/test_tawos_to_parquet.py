import datetime as dt

import pyarrow as pa
import pytest

from erp.extract.tawos_to_parquet import arrow_type, clean_datetimes


@pytest.mark.parametrize(
    ("mysql_type", "expected"),
    [
        ("int", pa.int64()),
        ("INT", pa.int64()),
        ("tinyint", pa.int8()),
        ("double", pa.float64()),
        ("datetime", pa.timestamp("s")),
        ("varchar", pa.large_string()),
        ("mediumtext", pa.large_string()),
    ],
)
def test_arrow_type_covers_every_type_in_the_tawos_schema(mysql_type, expected):
    assert arrow_type(mysql_type) == expected


def test_arrow_type_rejects_unknown_types_instead_of_guessing():
    with pytest.raises(ValueError, match="blob"):
        arrow_type("blob")


def test_clean_datetimes_keeps_valid_values_and_nulls():
    when = dt.datetime(2020, 3, 1, 9, 30)
    assert clean_datetimes([when, None]) == ([when, None], 0)


def test_clean_datetimes_nulls_unparseable_zero_dates():
    when = dt.datetime(2020, 3, 1)
    cleaned, replaced = clean_datetimes([when, "0000-00-00 00:00:00", None])
    assert cleaned == [when, None, None]
    assert replaced == 1
