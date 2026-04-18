from backend.app.services.transcript_parser import (
    format_timestamp,
    parse_timestamp,
    parse_transcript,
)


def test_parse_timestamp():
    assert parse_timestamp("00:00") == 0
    assert parse_timestamp("01:30") == 90
    assert parse_timestamp("10:05") == 605


def test_format_timestamp():
    assert format_timestamp(0) == "[00:00]"
    assert format_timestamp(90) == "[01:30]"
    assert format_timestamp(605) == "[10:05]"


def test_parse_transcript_basic():
    raw = (
        "[00:00] Hello world\n"
        "[01:15] Second line\n"
        "[02:30] Third line\n"
    )
    lines = parse_transcript(raw, "ep_test")
    assert len(lines) == 3
    assert lines[0].text == "Hello world"
    assert lines[0].start_seconds == 0
    assert lines[0].episode_id == "ep_test"
    assert lines[1].start_seconds == 75
    assert lines[2].start_seconds == 150


def test_parse_transcript_skips_blank_and_malformed():
    raw = (
        "[00:00] Valid line\n"
        "\n"
        "no timestamp here\n"
        "[01:00] Another valid\n"
    )
    lines = parse_transcript(raw, "ep_test")
    assert len(lines) == 2
    assert lines[0].text == "Valid line"
    assert lines[1].text == "Another valid"
