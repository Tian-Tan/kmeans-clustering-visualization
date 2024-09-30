# Makefile for Flask KMeans Clustering Web Application

# Python interpreter
PYTHON := python3

# Virtual environment
VENV := venv
VENV_ACTIVATE := $(VENV)/bin/activate

# Flask application file
APP := app.py

# Port to run the application on
PORT := 3000

# Install dependencies
install:
	$(PYTHON) -m venv $(VENV)
	. $(VENV_ACTIVATE) && pip install -r requirements.txt

# Run the Flask application
run:
	python app.py

# Clean up
clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

.PHONY: install run clean