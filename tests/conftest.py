"""Pytest hooks that write a test-result report at the end of the suite run.

The report (reports/test_report.txt by default) lists every test case with its
outcome and duration, followed by the full stack trace of every failure or
error. Override the location with the TEST_REPORT_PATH environment variable.
"""

import os
import time
from collections import Counter

import pytest

REPORT_PATH = os.environ.get("TEST_REPORT_PATH", os.path.join("reports", "test_report.txt"))

_results = []  # list of (nodeid, outcome, duration, longrepr_text)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    # Record the "call" phase for normal outcomes; also record setup/teardown
    # phases when they fail so errors are not lost.
    if report.when == "call" or report.failed:
        longrepr = str(report.longrepr) if report.failed else ""
        status = report.outcome
        if hasattr(report, "wasxfail"):
            status = "xfailed" if report.skipped else "xpassed"
        elif report.failed and report.when != "call":
            status = "error"
        _results.append((report.nodeid, status, report.duration, longrepr))


def pytest_sessionfinish(session, exitstatus):
    os.makedirs(os.path.dirname(REPORT_PATH) or ".", exist_ok=True)

    counts = Counter(status for _, status, _, _ in _results)
    failures = [(n, s, lr) for n, s, _, lr in _results if s in ("failed", "error")]

    lines = []
    lines.append("=" * 78)
    lines.append("TEST RESULT REPORT")
    lines.append(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Exit code : {int(exitstatus)}")
    lines.append(f"Total     : {len(_results)}")
    for status in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"):
        if counts.get(status):
            lines.append(f"{status.capitalize():<10}: {counts[status]}")
    lines.append("=" * 78)
    lines.append("")
    lines.append("TEST CASES")
    lines.append("-" * 78)
    for nodeid, status, duration, _ in _results:
        lines.append(f"[{status.upper():<7}] {duration:7.3f}s  {nodeid}")
    lines.append("")

    lines.append("STACK TRACES")
    lines.append("-" * 78)
    if not failures:
        lines.append("No failures or errors.")
    for nodeid, status, longrepr in failures:
        lines.append(f"--- {status.upper()}: {nodeid}")
        lines.append(longrepr)
        lines.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    terminalreporter.write_sep("=", f"test report written to {REPORT_PATH}")
