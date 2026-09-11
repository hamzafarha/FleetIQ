.PHONY: setup test lint clean run-pipeline api streamlit simulate

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

run-pipeline:
	python scripts/run_pipeline.py

api:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	streamlit run app_streamlit.py --server.port=8501

simulate:
	python simulate_stream.py --num-trips 15

test:
	pytest tests/ -v

lint:
	python -m py_compile $$(find src -name "*.py")

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

