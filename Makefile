.PHONY: setup test lint clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	python -m py_compile $$(find src -name "*.py")

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
