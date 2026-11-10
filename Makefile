.PHONY: install install-torch lint test check params

install:
	pip install -e ".[dev]"

install-torch:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev,onnx]"

lint:
	ruff check src tests

test:
	pytest

params:
	urbiva-ai params --width 1.0

check: lint test
