Typical operator flow you can now run
make install && make data — create venv, install, download v5.2.
python scripts/train_baseline.py --targets target target_cyrusd_20 target_teager2b_20 --params deep --seeds 0 1 2 3 --device cuda — seed-averaged deep-LightGBM baselines with walk-forward OOF; metrics logged to runs/.
Fit the MMCAwareStacker on the OOF matrix (from numerai_stack.stack), write weights to a YAML config.
python scripts/build_pickle.py --config configs/prod.yaml --out artifacts/predict.pkl — builds + smoke-tests the cloudpickle.
python scripts/submit.py --pickle artifacts/predict.pkl --model-id <uuid> — uploads, waits for VALIDATED, assigns.
Weekly: python scripts/weekly_report.py --models <id1> <id2> ... — generates a markdown report flagging MMC regressions; feed the round-payout history into optimize_stakes to rebalance.
Related past session: Numerai stack build session.