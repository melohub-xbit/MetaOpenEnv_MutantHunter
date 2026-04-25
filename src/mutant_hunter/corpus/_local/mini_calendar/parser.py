"""Proleptic Gregorian calendar helpers.

Pure Python, no `datetime` dependency. Working in raw ``(year, month, day)``
integer tuples is the point — it gives mutation testing a wide surface of
arithmetic, comparisons, and bounds to bite into.

Conventions:
    * ``Date`` is ``(year, month, day)`` with ``1 <= year <= 9999``,
      ``1 <= month <= 12``, ``1 <= day <= days_in_month(year, month)``.
    * Day-of-week is Monday=0..Sunday=6 (matches ``datetime.date.weekday``).
    * "Julian day" here means the proleptic Gregorian day count starting
      at 1 for ``(1, 1, 1)``. It is *not* the astronomical Julian Day Number.
    * ISO 8601 weeks (Monday-start, week 1 contains the first Thursday).
"""

from __future__ import annotations

from typing import Tuple

Date = Tuple[int, int, int]

MIN_YEAR = 1
MAX_YEAR = 9999

# Days per month in normal and leap years.
_MONTH_DAYS_NORMAL: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_MONTH_DAYS_LEAP: tuple[int, ...] = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# Cumulative days BEFORE the given month index. _CUM_DAYS_X[m-1] gives the
# number of days in months 1..(m-1), so day-of-year for (Y, m, d) is
# _CUM_DAYS_X[m-1] + d.
_CUM_DAYS_NORMAL: tuple[int, ...] = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
_CUM_DAYS_LEAP: tuple[int, ...] = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)

WEEKDAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #


def is_leap_year(year: int) -> bool:
    """Return True iff ``year`` is a Gregorian leap year.

    A year is a leap year iff it is divisible by 4, except centuries that
    are not divisible by 400. So 2000 is a leap year, 1900 is not.
    """
    if year % 4 != 0:
        return False
    if year % 100 != 0:
        return True
    return year % 400 == 0


def days_in_month(year: int, month: int) -> int:
    """Return the number of days in ``(year, month)``.

    Raises ``ValueError`` when ``month`` is outside ``1..12``.
    """
    if month < 1 or month > 12:
        raise ValueError(f"month must be in 1..12, got {month}")
    if month == 2 and is_leap_year(year):
        return 29
    return _MONTH_DAYS_NORMAL[month - 1]


def is_valid_date(year: int, month: int, day: int) -> bool:
    """Return True iff ``(year, month, day)`` is a valid in-range Gregorian date."""
    if year < MIN_YEAR or year > MAX_YEAR:
        return False
    if month < 1 or month > 12:
        return False
    if day < 1 or day > days_in_month(year, month):
        return False
    return True


def _require_valid(year: int, month: int, day: int) -> None:
    if not is_valid_date(year, month, day):
        raise ValueError(f"invalid date: {year:04d}-{month:02d}-{day:02d}")


# --------------------------------------------------------------------------- #
# Parsing / formatting                                                        #
# --------------------------------------------------------------------------- #


def parse_iso_date(s: str) -> Date:
    """Parse a canonical ISO 8601 calendar date ``YYYY-MM-DD``.

    Accepts only the 10-character form. The function does not strip
    whitespace; callers should normalise the input first.
    """
    if not isinstance(s, str):
        raise TypeError(f"parse_iso_date expects str, got {type(s).__name__}")
    if len(s) != 10:
        raise ValueError(f"expected 10 chars (YYYY-MM-DD), got {len(s)}")
    if s[4] != "-" or s[7] != "-":
        raise ValueError(f"expected dashes at positions 4 and 7, got {s!r}")
    try:
        year = int(s[0:4])
        month = int(s[5:7])
        day = int(s[8:10])
    except ValueError as exc:
        raise ValueError(f"non-numeric components in {s!r}") from exc
    _require_valid(year, month, day)
    return (year, month, day)


def format_iso(year: int, month: int, day: int) -> str:
    """Format ``(year, month, day)`` as ``YYYY-MM-DD``."""
    _require_valid(year, month, day)
    return f"{year:04d}-{month:02d}-{day:02d}"


# --------------------------------------------------------------------------- #
# Day-of-year conversions                                                     #
# --------------------------------------------------------------------------- #


def ordinal_day(year: int, month: int, day: int) -> int:
    """Return the 1-indexed day of year for ``(year, month, day)``.

    Examples: ``ordinal_day(2024, 1, 1) == 1``,
    ``ordinal_day(2024, 3, 1) == 61`` (leap),
    ``ordinal_day(2023, 3, 1) == 60`` (non-leap).
    """
    _require_valid(year, month, day)
    cum = _CUM_DAYS_LEAP if is_leap_year(year) else _CUM_DAYS_NORMAL
    return cum[month - 1] + day


def from_ordinal(year: int, ordinal: int) -> Date:
    """Inverse of :func:`ordinal_day`.

    Given ``(year, day_of_year)``, return the ``(Y, M, D)`` triple.
    """
    if year < MIN_YEAR or year > MAX_YEAR:
        raise ValueError(f"year out of range: {year}")
    days_in_year = 366 if is_leap_year(year) else 365
    if ordinal < 1 or ordinal > days_in_year:
        raise ValueError(f"ordinal {ordinal} outside 1..{days_in_year} for year {year}")
    cum = _CUM_DAYS_LEAP if is_leap_year(year) else _CUM_DAYS_NORMAL
    # Walk months down from December until the cumulative days-before-month
    # is strictly less than the ordinal — that's the month we're in.
    month = 12
    while month > 1 and cum[month - 1] >= ordinal:
        month -= 1
    day = ordinal - cum[month - 1]
    return (year, month, day)


# --------------------------------------------------------------------------- #
# Julian day arithmetic                                                       #
# --------------------------------------------------------------------------- #


def to_julian_day(year: int, month: int, day: int) -> int:
    """Days since the proleptic-Gregorian epoch ``(1, 1, 1)`` (which has JD 1).

    Used as the canonical integer representation for date arithmetic. The
    function is monotonic: ``to_julian_day(d2) > to_julian_day(d1)`` iff
    ``d2`` is strictly after ``d1``.
    """
    _require_valid(year, month, day)
    y = year - 1
    days = y * 365 + y // 4 - y // 100 + y // 400
    return days + ordinal_day(year, month, day)


def from_julian_day(jd: int) -> Date:
    """Inverse of :func:`to_julian_day`."""
    if jd < 1:
        raise ValueError(f"julian day must be >= 1, got {jd}")
    # 400-year cycle has exactly 146097 days. Jump forward by whole cycles
    # before searching year-by-year inside the residual.
    n400, jd_mod = divmod(jd - 1, 146097)
    year = n400 * 400 + 1
    while True:
        days_y = 366 if is_leap_year(year) else 365
        if jd_mod < days_y:
            return from_ordinal(year, jd_mod + 1)
        jd_mod -= days_y
        year += 1


def add_days(year: int, month: int, day: int, delta: int) -> Date:
    """Return the date ``delta`` days after ``(year, month, day)``.

    ``delta`` may be negative or zero. The result is validated to lie
    within the supported year range.
    """
    return from_julian_day(to_julian_day(year, month, day) + delta)


def date_diff(d1: Date, d2: Date) -> int:
    """Signed day difference: ``to_julian_day(d2) - to_julian_day(d1)``."""
    return to_julian_day(*d2) - to_julian_day(*d1)


# --------------------------------------------------------------------------- #
# Day of week                                                                 #
# --------------------------------------------------------------------------- #


def day_of_week(year: int, month: int, day: int) -> int:
    """Return the day of week as Monday=0..Sunday=6.

    Implemented via Zeller's congruence (Gregorian variant) so that it
    cross-checks the Julian-day path — mutating either pathway in
    isolation will diverge them.
    """
    _require_valid(year, month, day)
    m = month
    y = year
    if m < 3:
        m += 12
        y -= 1
    K = y % 100
    J = y // 100
    # Zeller's `h`: 0=Saturday, 1=Sunday, 2=Monday, ..., 6=Friday.
    h = (day + (13 * (m + 1)) // 5 + K + K // 4 + J // 4 + 5 * J) % 7
    # Convert to Monday=0..Sunday=6.
    return (h + 5) % 7


def weekday_name(year: int, month: int, day: int) -> str:
    """Return the three-letter English weekday name."""
    return WEEKDAYS[day_of_week(year, month, day)]


# --------------------------------------------------------------------------- #
# Business days                                                               #
# --------------------------------------------------------------------------- #


def is_weekend(year: int, month: int, day: int) -> bool:
    """True iff the date falls on a Saturday or Sunday."""
    return day_of_week(year, month, day) >= 5


def is_business_day(year: int, month: int, day: int) -> bool:
    """True iff the date is Monday-through-Friday."""
    return not is_weekend(year, month, day)


def next_business_day(year: int, month: int, day: int) -> Date:
    """Return the first business day strictly after ``(year, month, day)``."""
    d = add_days(year, month, day, 1)
    while is_weekend(*d):
        d = add_days(*d, 1)
    return d


def previous_business_day(year: int, month: int, day: int) -> Date:
    """Return the last business day strictly before ``(year, month, day)``."""
    d = add_days(year, month, day, -1)
    while is_weekend(*d):
        d = add_days(*d, -1)
    return d


def business_days_between(d1: Date, d2: Date) -> int:
    """Count business days strictly between ``d1`` (exclusive) and ``d2`` (inclusive).

    Negative when ``d2`` is before ``d1``. Zero when ``d1 == d2``.
    """
    diff = date_diff(d1, d2)
    if diff == 0:
        return 0
    step = 1 if diff > 0 else -1
    count = 0
    cur = d1
    for _ in range(abs(diff)):
        cur = add_days(*cur, step)
        if is_business_day(*cur):
            count += step
    return count


# --------------------------------------------------------------------------- #
# ISO week number                                                             #
# --------------------------------------------------------------------------- #


def iso_week_number(year: int, month: int, day: int) -> int:
    """Return the ISO 8601 week number (1..53) for ``(year, month, day)``.

    A week's "ISO year" is determined by the year of its Thursday.
    Week 1 of an ISO year is the week containing January 4 of that year
    (equivalently, the first week with at least 4 days in the new year).
    """
    _require_valid(year, month, day)
    target_jd = to_julian_day(year, month, day)
    dow = day_of_week(year, month, day)
    # Thursday in the same ISO week:
    thursday_jd = target_jd + (3 - dow)
    thursday_year, _, _ = from_julian_day(thursday_jd)
    # Week 1 of `thursday_year` contains its January 4:
    jan4_jd = to_julian_day(thursday_year, 1, 4)
    jan4_dow = day_of_week(thursday_year, 1, 4)
    week1_monday_jd = jan4_jd - jan4_dow
    monday_jd = target_jd - dow
    return (monday_jd - week1_monday_jd) // 7 + 1


__all__ = [
    "Date",
    "MIN_YEAR",
    "MAX_YEAR",
    "WEEKDAYS",
    "add_days",
    "business_days_between",
    "date_diff",
    "day_of_week",
    "days_in_month",
    "format_iso",
    "from_julian_day",
    "from_ordinal",
    "is_business_day",
    "is_leap_year",
    "is_valid_date",
    "is_weekend",
    "iso_week_number",
    "next_business_day",
    "ordinal_day",
    "parse_iso_date",
    "previous_business_day",
    "to_julian_day",
    "weekday_name",
]
