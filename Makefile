.PHONY: all

# https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONDA_ENV_NAME=rele-zoo
PROJECT_PATH=relezoo

install-env:
	conda env create -f environment.yaml

install:
	pip install -e .

update-env:
	conda env update -n rele-zoo -f environment.yaml


clean: clean-pyc clean-test clean-misc

clean-misc:
	rm -rf .benchmarks
	rm report.xml || true
	rm coverage.xml || true
	rm .coverage || true
	rm -rf outputs || true
	rm -rf relezoo.egg-info || true
	rm -rf .pytest_cache || true
	rm -rf tensorboard || true
	rm -rf checkpoints || true

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -rf .coverage
	rm -rf .pytest_cache
	rm -rf .mypy_cache

typing:
	($(CONDA_ACTIVATE) ${CONDA_ENV_NAME}; pytest -v -s --mypy ${PROJECT_PATH} tests)

lint:
	($(CONDA_ACTIVATE) ${CONDA_ENV_NAME}; flake8 --output-file=flake8.txt || true)

test:
	($(CONDA_ACTIVATE) ${CONDA_ENV_NAME}; pytest --junitxml=report.xml -v -s --cov=${PROJECT_PATH} tests; coverage xml)

benchmark:
	($(CONDA_ACTIVATE) ${CONDA_ENV_NAME}; pytest -v -s -m benchmark tests || true )

all: clean install-env test
