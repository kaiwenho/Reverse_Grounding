"""Shared test fixtures, and one guard against a silent environment fault.

The anchor-category check needs the Biolink model, which lives in plan-core and
is imported softly — the controller must be able to read a result document
without it. That softness made a real misconfiguration look like nine unrelated
assertion failures (`assert 0 == 1`, `assert 'absence' == 'inconclusive'`) with
nothing anywhere naming the cause.

`requires_biolink` turns those nine into nine copies of one sentence that says
what is missing and how to fix it. It fails rather than skips, deliberately: a
skipped test on a green run is how a check quietly stops being exercised.
"""

from __future__ import annotations

import pytest

from loop_controller.diagnose import biolink_check_status


@pytest.fixture
def requires_biolink():
    """Fail with the reason when the Biolink model cannot be loaded."""
    test, reason = biolink_check_status()
    if test is None:
        pytest.fail(
            f"the Biolink model is not loadable, so the anchor-category check "
            f"cannot run and every anchor looks sound: {reason}. Run the suite "
            f"with plan-core on the path:\n"
            f"    PYTHONPATH=../plan-core/src:src python -m pytest tests/ -q"
        )
    return test
