SEED ?= 0
ARM ?= ce
NORM ?= qsigmoid
MODE ?= mul
DEVICE ?= mps
PYTHON ?= DAEDL/.venv/bin/python
export PYTHONPATH := src:$(PYTHONPATH)
export TRANSFORMERS_OFFLINE ?= 1
export HF_HUB_OFFLINE ?= 1

test:
	$(PYTHON) -m pytest -q -m "not slow"

test-all:
	$(PYTHON) -m pytest -q

ce:
	$(PYTHON) scripts/run_experiment.py --arm ce --seed $(SEED) --device $(DEVICE)

edl:
	$(PYTHON) scripts/run_experiment.py --arm edl --seed $(SEED) --device $(DEVICE)

daedl:
	$(PYTHON) scripts/run_experiment.py --arm daedl --seed $(SEED) --device $(DEVICE) \
		--normalizer $(NORM) --combine_mode $(MODE)

seeds:
	for s in 0 1 2 3 4; do \
		$(PYTHON) scripts/run_experiment.py --arm $(ARM) --seed $$s --device $(DEVICE); \
	done

aggregate:
	$(PYTHON) scripts/aggregate_results.py results/

.PHONY: test test-all ce edl daedl seeds aggregate
