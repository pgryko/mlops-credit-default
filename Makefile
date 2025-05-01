#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = credit-default-prediction
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip uv
	uv pip install -e .

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 creditrisk

all:
	requirements clean lint

## Download data from Kaggle
.PHONY: download
download:
	python -m creditrisk.data.preproc


.PHONY: preprocess
preprocess:
	python -m creditrisk.data.preproc

.PHONY: train
train:
	python -m creditrisk.models.train

.PHONY: resolve
resolve:
	python -m creditrisk.models.resolve

.PHONY: predict
predict:
	python -m creditrisk.models.predict