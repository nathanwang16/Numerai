.PHONY: install test data train baseline smoke

PY?=.venv/bin/python
PYTEST?=.venv/bin/pytest

install:
	python3.12 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cu124
	$(PY) -m pip install pytest

test:
	$(PYTEST) tests/ -v

data:
	$(PY) scripts/download_data.py --cache-dir data

baseline:
	$(PY) scripts/train_baseline.py --targets target target_cyrusd_20 target_teager2b_20 \
		--params deep --seeds 0 1 2 3 --device cpu --every-nth-era 4

baseline-fast:
	$(PY) scripts/train_baseline.py --targets target --params standard --seeds 0 1 \
		--every-nth-era 8 --dry-run

smoke:
	$(PY) -c "import numerai_stack; print('package OK', numerai_stack.__version__)"
	$(PYTEST) tests/ -q

# Reproduce a specific run by id
reproduce:
	@test -n "$$RUN" || (echo "set RUN=<run_id>" && exit 1)
	$(PY) scripts/train_baseline.py --config runs/$$RUN/config.yaml
