"""Python scorers for benchmarks/structured_output.yaml.

A scorer is a function ``(response, case) -> bool | dict``: ``response`` is the
model's reply as text, ``case`` the test case (``id``, ``variables``,
``expected_output``, ``tags``…). Return ``True``/``False``, or
``{"passed": bool, "score": 0–1, "reason": str}`` to grade it and say why.
An exception counts as a pipeline error (``scorer error``), not a failed case.
"""

import json


def short_summary(response, case):
    """The reply is JSON whose summary is a single sentence of at most 25 words."""
    try:
        summary = json.loads(response.strip().strip("`").removeprefix("json"))[
            "summary"
        ]
    except (ValueError, KeyError, TypeError):
        return {"passed": False, "score": 0.0, "reason": "no JSON summary"}
    words = len(str(summary).split())
    sentences = sum(str(summary).count(p) for p in ".!?") or 1
    passed = words <= 25 and sentences == 1
    return {
        "passed": passed,
        "score": min(1.0, 25 / max(words, 1)),
        "reason": f"{words} words",
    }
