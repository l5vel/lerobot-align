"""Shared, fail-closed eligibility rules for both contrast producers."""

from __future__ import annotations


def contrast_policy(arm: str, baseline: str, metadata: dict,
                    *, allow_unequalised: bool = False,
                    allow_reference: bool = False) -> tuple[str | None, str | None]:
    for name in (arm, baseline):
        meta = metadata.get(name)
        if not meta or not meta.get("supervision") or not meta.get("tool"):
            return f"missing metadata or supervision for '{name}'", None
        if meta.get("quarantined"):
            return f"quarantined: '{name}' consumes ground truth; upper bound only", None
        if meta.get("equalised") is False and not allow_unequalised:
            return f"not equalised: '{name}' sees a different camera set", None
    a, b = metadata[arm], metadata[baseline]
    if a["supervision"] != b["supervision"]:
        if allow_reference and (a["tool"] == "reference" or b["tool"] == "reference"):
            return None, (
                f"Supervision asymmetry: {arm} uses '{a['supervision']}', while "
                f"{baseline} uses '{b['supervision']}'. The model-free seed-calibrated "
                "reference sees labelled seed episodes and no video; VLM arms, when "
                "present, see video without those seed labels. This is a floor comparison, not a "
                "like-for-like tool effect."
            )
        return (f"supervision mismatch: '{arm}' uses '{a['supervision']}', "
                f"'{baseline}' uses '{b['supervision']}'"), None
    return None, None
