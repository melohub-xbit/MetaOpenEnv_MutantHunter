"""Existing test suite for ``mini_calendar.parser``.

DELIBERATELY WEAK. This is a mutation-testing target: the entire purpose of
the suite below is to leave plenty of surviving mutants for an RL agent to
catch by writing additional tests. It covers only happy paths on a handful
of mid-month dates and skips:

    * Leap-year century rules (1900, 2000) and February 29.
    * Month / year rollovers in :func:`add_days`.
    * Negative or zero ``delta``.
    * Off-by-one bounds on month and ordinal validation.
    * Business-day rollover across weekends.
    * ISO week numbers at year boundaries (Dec 28-31 / Jan 1-3).
    * Inverse properties (``parse_iso_date(format_iso(d)) == d``, etc.).
    * Error paths.
"""

from mini_calendar.parser import (
    add_days,
    business_days_between,
    date_diff,
    day_of_week,
    days_in_month,
    format_iso,
    is_business_day,
    is_leap_year,
    is_weekend,
    iso_week_number,
    next_business_day,
    ordinal_day,
    parse_iso_date,
    weekday_name,
)


def test_parse_iso_date_basic():
    assert parse_iso_date("2024-06-15") == (2024, 6, 15)


def test_format_iso_basic():
    assert format_iso(2024, 6, 15) == "2024-06-15"


def test_is_leap_year_2024():
    assert is_leap_year(2024) is True


def test_is_leap_year_2023():
    assert is_leap_year(2023) is False


def test_days_in_january():
    assert days_in_month(2024, 1) == 31


def test_days_in_april():
    assert days_in_month(2024, 4) == 30


def test_day_of_week_known_saturday():
    # 2024-06-15 was a Saturday (Mon=0..Sun=6 → 5).
    assert day_of_week(2024, 6, 15) == 5


def test_weekday_name_known():
    assert weekday_name(2024, 6, 15) == "Sat"


def test_add_days_simple_positive():
    assert add_days(2024, 6, 15, 10) == (2024, 6, 25)


def test_date_diff_simple():
    assert date_diff((2024, 6, 15), (2024, 6, 25)) == 10


def test_ordinal_day_january_first():
    assert ordinal_day(2024, 1, 1) == 1


def test_is_weekend_saturday():
    assert is_weekend(2024, 6, 15) is True


def test_is_business_day_wednesday():
    assert is_business_day(2024, 6, 12) is True


def test_next_business_day_from_wednesday():
    # 2024-06-12 is a Wednesday → next business day is Thursday 2024-06-13.
    assert next_business_day(2024, 6, 12) == (2024, 6, 13)


def test_business_days_between_within_week():
    # 2024-06-17 (Mon, exclusive) to 2024-06-21 (Fri, inclusive):
    # business days counted = Tue, Wed, Thu, Fri → 4.
    assert business_days_between((2024, 6, 17), (2024, 6, 21)) == 4


def test_iso_week_number_mid_year():
    assert iso_week_number(2024, 6, 15) == 24
