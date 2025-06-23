# Determine the operating system
ifeq ($(OS),Windows_NT)
    # Windows
    PYTHON = py
    RM = del /Q
else
    # macOS and Linux
    PYTHON = python3
    RM = rm -f
endif

# Create virtual environment (only if it doesn't exist)
venv:
	$(PYTHON) -m venv .venv

install:
	pip install -e .

docs:
	pip install mkdocs "mkdocstrings[python]" mkdocs-material
	mkdocs gh-deploy

build:
	pip install build
	python -m build

# Run tests
test:
	python -m unittest discover -s tests -f

test_window:
	python -m unittest tests/test_window.py

test_eplus:
	python -m unittest tests/test_eplus.py

# Clean up
clean:
	$(RM) *.pyc
	$(RM) -r __pycache__
	$(RM) -r build
	$(RM) -r dist
	$(RM) -r *.egg-info

deep-clean: clean
	$(RM) -r .venv

.PHONY: venv install build test clean docs test_window test_eplus
