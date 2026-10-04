.PHONY: install install-gpu install-dev test lint format

install:            ## Основные зависимости + пакет peft_lab в режиме разработки
	pip install -r requirements.txt
	pip install -e . --no-deps

install-gpu:        ## То же + bitsandbytes и Unsloth (Linux + NVIDIA)
	pip install -r requirements-gpu.txt
	pip install -e . --no-deps

install-dev:        ## Линтер и тесты
	pip install -r requirements-dev.txt

test:
	pytest -m "not gpu and not slow"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

