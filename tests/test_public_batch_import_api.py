from __future__ import annotations

from typing import get_args

import discourselens
from discourselens import BatchImportItemResult, BatchImportResult, BatchImportStatus
from discourselens.documents import (
    BatchImportItemResult as DocumentBatchImportItemResult,
)
from discourselens.documents import BatchImportResult as DocumentBatchImportResult
from discourselens.documents import BatchImportStatus as DocumentBatchImportStatus


def test_batch_import_contracts_are_available_from_public_package():
    assert BatchImportItemResult is DocumentBatchImportItemResult
    assert BatchImportResult is DocumentBatchImportResult
    assert BatchImportStatus is DocumentBatchImportStatus
    assert get_args(BatchImportStatus) == ("SUCCESS", "DUPLICATE", "FAILED")
    assert {
        "BatchImportItemResult",
        "BatchImportResult",
        "BatchImportStatus",
    } <= set(discourselens.__all__)
