"""Upload / assign / trigger pickles for Numerai Compute.

We use the ``numerapi`` client directly (it wraps the same GraphQL endpoints
as the MCP ``upload_model`` tool) because it is fully scriptable from Python
and works offline-friendly for the rest of the stack. Credentials come from:

    NUMERAI_PUBLIC_ID + NUMERAI_SECRET_KEY env vars
    or explicit (public_id, secret_key) args

For interactive / ad-hoc work, the ``upload_model`` MCP tool in
``/home/lemon/.cursor/projects/home-lemon-Numerai/mcps/user-numerai`` is
equivalent and can be preferred when the user wants Cursor to drive the flow.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from .credentials import get_credentials


def _client(public_id: str | None = None, secret_key: str | None = None):
    from numerapi import NumerAPI

    creds = get_credentials(public_id, secret_key)
    return NumerAPI(public_id=creds.public_id, secret_key=creds.secret_key)


def upload_pickle(
    path: str | Path,
    model_id: str,
    tournament: int = 8,
    public_id: str | None = None,
    secret_key: str | None = None,
) -> str:
    """Upload a pickle file and return the pickle UUID.

    The returned UUID is what ``assign_pickle`` consumes.
    """
    napi = _client(public_id, secret_key)
    pickle_id = napi.model_upload(
        file_path=str(path),
        model_id=model_id,
        tournament=tournament,
    )
    return pickle_id


def list_pickles(
    model_id: str,
    public_id: str | None = None,
    secret_key: str | None = None,
    unassigned: bool = False,
) -> list[dict[str, Any]]:
    """Return Numerai Compute pickles for the given model.

    Each entry is the raw ``ComputePickleUpload`` GraphQL object (id,
    filename, validationStatus, diagnosticsStatus, triggerStatus, label,
    version, insertedAt, updatedAt, modelId).
    """
    napi = _client(public_id, secret_key)
    query = """
      query($modelId: String!, $unassigned: Boolean!) {
        computePickles(modelId: $modelId, unassigned: $unassigned) {
          id
          filename
          label
          version
          modelId
          validationStatus
          diagnosticsStatus
          triggerStatus
          insertedAt
          updatedAt
        }
      }
    """
    data = napi.raw_query(
        query,
        variables={"modelId": model_id, "unassigned": unassigned},
        authorization=True,
    )
    return data["data"]["computePickles"] or []


def assign_pickle(
    pickle_id: str,
    model_id: str,
    public_id: str | None = None,
    secret_key: str | None = None,
) -> dict[str, Any]:
    """Bind ``pickle_id`` to ``model_id``. Returns the raw GraphQL response."""
    napi = _client(public_id, secret_key)
    query = """
      mutation($pickleId: String!, $modelId: String!) {
        assignPickleToModel(pickleId: $pickleId, modelId: $modelId)
      }
    """
    return napi.raw_query(
        query,
        variables={"pickleId": pickle_id, "modelId": model_id},
        authorization=True,
    )


def trigger_submission(
    pickle_id: str,
    model_id: str,
    trigger_validation: bool = False,
    public_id: str | None = None,
    secret_key: str | None = None,
) -> dict[str, Any]:
    """Run an assigned pickle for the current round (or as a validation run)."""
    napi = _client(public_id, secret_key)
    query = """
      mutation($pickleId: String!, $modelId: String!, $triggerValidation: Boolean!) {
        triggerComputePickleUpload(
          pickleId: $pickleId,
          modelId: $modelId,
          triggerValidation: $triggerValidation
        ) {
          id
          triggerStatus
        }
      }
    """
    return napi.raw_query(
        query,
        variables={
            "pickleId": pickle_id,
            "modelId": model_id,
            "triggerValidation": trigger_validation,
        },
        authorization=True,
    )


# Status vocabulary returned by the current Numerai schema (lowercase).
_VALIDATION_DONE = {"passed", "ready", "validated", "completed", "success"}
_VALIDATION_FAILED = {"failed", "error"}


def wait_for_pickle_ready(
    pickle_id: str,
    model_id: str,
    timeout_seconds: int = 1200,
    poll_seconds: int = 20,
    public_id: str | None = None,
    secret_key: str | None = None,
) -> str:
    """Poll the pickle's ``validationStatus`` until done (or fails/timeouts).

    Returns the final status string (lowercase). Raises ``RuntimeError`` on a
    failure status or ``TimeoutError`` if the deadline is hit while still in
    a non-terminal state (e.g. ``validating``).
    """
    start = time.time()
    last_status: str | None = None
    while time.time() - start < timeout_seconds:
        pickles = list_pickles(model_id, public_id=public_id, secret_key=secret_key)
        for pk in pickles:
            if pk.get("id") == pickle_id:
                status = (pk.get("validationStatus") or "").lower()
                last_status = status
                if status in _VALIDATION_DONE:
                    return status
                if status in _VALIDATION_FAILED:
                    raise RuntimeError(f"Pickle {pickle_id} validationStatus={status!r}")
                break
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"Pickle {pickle_id} not ready in {timeout_seconds}s (last status={last_status!r})"
    )


__all__ = [
    "upload_pickle",
    "list_pickles",
    "assign_pickle",
    "trigger_submission",
    "wait_for_pickle_ready",
]
