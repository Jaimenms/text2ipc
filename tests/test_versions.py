import datetime as dt

from text2ipc import versions

HTML = """
<a href="/classifications/data/ipc/ITSupport_and_download_area/20250101/">20250101</a>
<a href="/classifications/data/ipc/ITSupport_and_download_area/20260101/">20260101</a>
<a href="/classifications/data/ipc/ITSupport_and_download_area/20270101/">20270101</a>
<a href="/classifications/data/ipc/ITSupport_and_download_area/Documentation/">docs</a>
<a href="/x/99999999/">bogus</a>
"""


def test_parse_version_index():
    assert versions.parse_version_index(HTML) == ["20250101", "20260101", "20270101"]


def test_resolve_latest_and_current(monkeypatch):
    monkeypatch.setattr(
        versions, "list_remote_versions", lambda: ["20250101", "20260101", "20270101"]
    )
    monkeypatch.setattr(versions, "list_local_versions", lambda root=None: [])
    assert versions.resolve_version("latest") == "20270101"
    assert versions.resolve_version("current", today=dt.date(2026, 9, 9)) == "20260101"
    assert versions.resolve_version("20240101") == "20240101"
