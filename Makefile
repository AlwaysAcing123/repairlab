.PHONY: test integration-test admit admit-splits demo notebook

test:
	python3 -B -m unittest discover -s tests -v

integration-test:
	REPAIRLAB_DOCKER_TESTS=1 python3 -B -m unittest discover -s tests -v

admit:
	python3 -B -m repairlab admit --output artifacts/admission-$$(date -u +%Y%m%dT%H%M%S)

admit-splits:
	python3 -B scripts/admit_splits.py --output artifacts/split-admission-$$(date -u +%Y%m%dT%H%M%S)

demo:
	python3 -B -m repairlab demo --output artifacts/demo-$$(date -u +%Y%m%dT%H%M%S)

notebook:
	python3 -B scripts/build_baseline_notebook.py
