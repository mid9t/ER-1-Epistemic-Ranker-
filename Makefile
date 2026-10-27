SEED ?= 0
ARM ?= ce
NORM ?= qsigmoid
MODE ?= mul
DEVICE ?= mps
ifeq ($(PYTHON),)
  PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; elif [ -x env/bin/python ]; then echo env/bin/python; else echo python3; fi)
endif
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

cache:
	for s in 0 1 2 3 4; do \
		$(PYTHON) scripts/run_experiment.py --arm $(ARM) --seed $$s --device $(DEVICE) \
			--save_cache --save_pooled; \
	done

scores:
	$(PYTHON) scripts/score_table.py --root results --out results/posthoc/scores --verify

grid:
	$(PYTHON) scripts/coupling_grid.py --root results --out results/posthoc/grid

fusion:
	$(PYTHON) scripts/complementarity.py --root results --out results/posthoc/complementarity

bootstrap:
	$(PYTHON) scripts/paired_bootstrap.py --preset item5 --root results \
		--selected results/posthoc/grid/selected.json \
		--out results/posthoc/bootstrap/item5.csv --fpr

gda-ablation:
	$(PYTHON) scripts/gda_ablation.py --root results \
		--selected results/posthoc/grid/selected.json --out results/posthoc/gda
	$(PYTHON) scripts/coupling_grid.py --root results --fit_on train id_val \
		--out results/posthoc/grid_fit_source

ablate-nosn:
	for s in 0 1 2; do \
		$(PYTHON) scripts/run_experiment.py --arm edl --seed $$s --device $(DEVICE) \
			--sn_bottleneck off --tag nosn --save_cache --save_pooled \
			--output_dir results/ablations; \
	done

ablate-mse:
	for s in 0 1 2; do \
		$(PYTHON) scripts/run_experiment.py --arm edl --seed $$s --device $(DEVICE) \
			--edl_loss mse --kl_schedule constant --reg_param 5e-2 --tag mse \
			--save_cache --save_pooled --output_dir results/ablations; \
	done

ablation-posthoc:
	$(PYTHON) scripts/score_table.py --root results/ablations --tag nosn --out results/posthoc/ablation_nosn
	$(PYTHON) scripts/score_table.py --root results/ablations --tag mse --out results/posthoc/ablation_mse
	$(PYTHON) scripts/coupling_grid.py --root results/ablations --tag nosn \
		--apply-selected results/posthoc/grid/selected.json --out results/posthoc/ablation_nosn/coupling
	$(PYTHON) scripts/coupling_grid.py --root results/ablations --tag mse \
		--apply-selected results/posthoc/grid/selected.json --out results/posthoc/ablation_mse/coupling
	$(PYTHON) scripts/paired_bootstrap.py --preset item7 --root results \
		--ablation-root results/ablations --selected results/posthoc/grid/selected.json \
		--out results/posthoc/bootstrap/item7.csv

cifar-ood:
	$(PYTHON) scripts/cifar_ood_side.py extract --daedl_dir DAEDL --ckpt saved_results/model.pt \
		--device $(DEVICE) --out results/cifar_ood_side
	$(PYTHON) scripts/cifar_ood_side.py analyse --out results/cifar_ood_side

b77-check:
	$(PYTHON) scripts/check_data.py --dataset banking77_oos --model_name bert-base-uncased \
		--max_length 64 --data_dir data/BANKING77-OOS --out results_b77/data_check.json

b77-train:
	for arm in ce edl; do for s in 0 1 2 3 4; do \
		$(PYTHON) scripts/run_experiment.py --dataset banking77_oos --arm $$arm --seed $$s \
			--device $(DEVICE) --save_cache --save_pooled \
			--data_dir data/BANKING77-OOS --output_dir results_b77; \
	done; done

b77-posthoc:
	PYTHON="$(PYTHON)" bash scripts/replication_posthoc.sh results_b77 all id ood

rbt-download:
	TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 $(PYTHON) -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('roberta-base'); AutoModel.from_pretrained('roberta-base')"

rbt-check:
	$(PYTHON) scripts/check_data.py --dataset clinc150 --model_name roberta-base \
		--max_length 64 --data_dir data/CLINC150 --out results_roberta/data_check.json

rbt-train:
	for arm in ce edl; do for s in 0 1 2 3 4; do \
		$(PYTHON) scripts/run_experiment.py --dataset clinc150 --model_name roberta-base \
			--arm $$arm --seed $$s --device $(DEVICE) --save_cache --save_pooled \
			--output_dir results_roberta; \
	done; done

rbt-posthoc:
	PYTHON="$(PYTHON)" bash scripts/replication_posthoc.sh results_roberta all

.PHONY: test test-all ce edl daedl seeds aggregate cache scores grid fusion bootstrap \
	gda-ablation ablate-nosn ablate-mse ablation-posthoc cifar-ood \
	b77-check b77-train b77-posthoc rbt-download rbt-check rbt-train rbt-posthoc
