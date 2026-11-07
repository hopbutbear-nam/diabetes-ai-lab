"""Gói an toàn: luật an toàn, phân tầng nguy cơ và miễn trừ trách nhiệm."""

from diabetes_ai.safety.critic import CheckResult, cross_check
from diabetes_ai.safety.disclaimers import (
    CONFIRM_TEST_ADVICE,
    DISCLAIMER_SHORT,
    DISCLAIMER_VI,
    EARLY_EXAM_ADVICE,
    advice_for,
)
from diabetes_ai.safety.risk_level import combine, level_from_index, risk_index
from diabetes_ai.safety.rules import Alert, Rule, evaluate, load_rules, top_level

__all__ = [
    "Alert",
    "CONFIRM_TEST_ADVICE",
    "CheckResult",
    "DISCLAIMER_SHORT",
    "DISCLAIMER_VI",
    "EARLY_EXAM_ADVICE",
    "Rule",
    "advice_for",
    "combine",
    "cross_check",
    "evaluate",
    "level_from_index",
    "load_rules",
    "risk_index",
    "top_level",
]
