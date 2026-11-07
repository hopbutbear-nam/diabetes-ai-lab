"""Gói giải thích mô hình."""

from diabetes_ai.explain.permutation import permutation_importance
from diabetes_ai.explain.shap_explain import explain_instance, shap_available
from diabetes_ai.explain.what_if import (
    WHAT_IF_NOTE,
    feature_effects,
    what_if_custom,
    what_if_scenarios,
)

__all__ = [
    "WHAT_IF_NOTE",
    "explain_instance",
    "feature_effects",
    "permutation_importance",
    "shap_available",
    "what_if_custom",
    "what_if_scenarios",
]
