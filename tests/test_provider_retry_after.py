from datetime import UTC, datetime, timedelta
from email.message import Message
from email.utils import format_datetime
from urllib.error import HTTPError

import pytest

from discourselens.providers import _retry_after_seconds


def _error(retry_after: str) -> HTTPError:
    headers = Message()
    headers["Retry-After"] = retry_after
    return HTTPError("https://provider.test", 429, "Too Many Requests", headers, None)


@pytest.mark.parametrize(("offset_seconds", "expected"), [(90, 90.0), (-30, 0.0)])
def test_retry_after_http_date_uses_controlled_utc_clock(offset_seconds, expected):
    now = datetime(2030, 10, 21, 7, 28, tzinfo=UTC)
    retry_at = now + timedelta(seconds=offset_seconds)

    delay = _retry_after_seconds(
        _error(format_datetime(retry_at, usegmt=True)),
        now=now,
    )

    assert delay == expected


@pytest.mark.parametrize(
    ("retry_after", "now", "expected"),
    [
        (
            "Monday, 21-Oct-30 07:29:30 GMT",
            datetime(2030, 10, 21, 7, 28, tzinfo=UTC),
            90.0,
        ),
        (
            "Wed Oct 21 07:29:30 2015",
            datetime(2015, 10, 21, 7, 28, tzinfo=UTC),
            90.0,
        ),
    ],
)
def test_retry_after_accepts_obsolete_http_date_forms(retry_after, now, expected):
    assert _retry_after_seconds(_error(retry_after), now=now) == expected


@pytest.mark.parametrize(
    "retry_after",
    [
        "not a date",
        "Wed, 31 Feb 2030 07:28:00 GMT",
        "Tue, 21 Oct 2030 07:28:00 GMT",
        "Mon, 21 Oct 2030 07:28:00 +0000",
        "Mon, 21 Oct 2030 07:28:00",
    ],
)
def test_retry_after_invalid_http_dates_return_none(retry_after):
    assert _retry_after_seconds(
        _error(retry_after), now=datetime(2030, 10, 21, 7, 28, tzinfo=UTC)
    ) is None


@pytest.mark.parametrize(("retry_after", "expected"), [("3.5", 3.5), (" 0 ", 0.0)])
def test_retry_after_numeric_seconds_remain_supported(retry_after, expected):
    assert _retry_after_seconds(_error(retry_after)) == expected
