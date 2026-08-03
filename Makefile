PYTHON ?= python3

.PHONY: all data bench figures spec clean

all: bench figures

data:
	$(PYTHON) src/data.py

bench:
	$(PYTHON) src/bench.py

figures:
	$(PYTHON) src/figures.py

spec:
	lean --run spec/RidgeTomography.lean
	cd spec && lake env lean Invariants.lean
	$(PYTHON) tools/check_spec.py

clean:
	rm -rf src/__pycache__ tools/__pycache__ results figures
