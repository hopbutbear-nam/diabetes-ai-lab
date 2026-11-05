"""Gói xử lý dữ liệu."""

from diabetes_ai.data.load import load_raw
from diabetes_ai.data.preprocess import (
    drop_exact_duplicates,
    duplicate_mask,
    encode_target,
)
from diabetes_ai.data.schema import (
    BINARY_CLASS_NAMES,
    BINARY_FEATURES,
    CLASSES,
    COUNT_FEATURES,
    FEATURE_RANGES,
    FEATURES,
    ORDINAL_FEATURES,
    TARGET,
    class_names,
)
from diabetes_ai.data.split import (
    SplitIndices,
    build_manifest,
    load_split,
    make_split,
    save_split,
    twin_mask,
    variant_name,
)
from diabetes_ai.data.validate import (
    ValidationReport,
    validate_dataframe,
    write_report,
)

__all__ = [
    "TARGET",
    "CLASSES",
    "FEATURES",
    "BINARY_FEATURES",
    "ORDINAL_FEATURES",
    "COUNT_FEATURES",
    "FEATURE_RANGES",
    "BINARY_CLASS_NAMES",
    "class_names",
    "load_raw",
    "ValidationReport",
    "validate_dataframe",
    "write_report",
    "encode_target",
    "duplicate_mask",
    "drop_exact_duplicates",
    "SplitIndices",
    "variant_name",
    "make_split",
    "twin_mask",
    "build_manifest",
    "save_split",
    "load_split",
]
