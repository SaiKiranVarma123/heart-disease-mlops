PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

download:
	$(PYTHON) scripts_download_data.py

eda:
	$(PYTHON) -m heart_disease.eda

train:
	$(PYTHON) -m heart_disease.train

test:
	pytest --cov=heart_disease --cov-report=term-missing

lint:
	ruff check src tests scripts_download_data.py

serve:
	uvicorn heart_disease.api:app --host 0.0.0.0 --port 8000
