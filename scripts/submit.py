"""Upload + assign a validated pickle to a Numerai Compute model slot.

Credentials read from env:
    NUMERAI_PUBLIC_ID
    NUMERAI_SECRET_KEY

Flow:
    1. ``numerai_stack.compute.submit.upload_pickle``  -- PUTs the .pkl
    2. poll ``list_pickles`` until VALIDATED
    3. ``assign_pickle`` -- attaches it to the model slot
"""
from __future__ import annotations

import argparse

from numerai_stack.compute import submit as compute_submit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--model-id", required=True, help="Numerai model UUID")
    ap.add_argument("--tournament", type=int, default=8)
    ap.add_argument("--no-assign", action="store_true",
                    help="Upload only, don't assign")
    args = ap.parse_args()

    print(f"uploading {args.pickle} -> model {args.model_id}")
    pickle_id = compute_submit.upload_pickle(
        path=args.pickle, model_id=args.model_id, tournament=args.tournament,
    )
    print(f"pickle_id = {pickle_id}")
    print("waiting for validation ...")
    status = compute_submit.wait_for_pickle_ready(pickle_id, args.model_id)
    print(f"status = {status}")
    if not args.no_assign:
        result = compute_submit.assign_pickle(pickle_id, args.model_id)
        print(f"assigned: {result}")


if __name__ == "__main__":
    main()
