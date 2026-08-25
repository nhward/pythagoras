PYTHON := .venv/bin/python
SHINY := .venv/bin/shiny
SHINYLIVE := .venv/bin/shinylive


APP_FILES := $(wildcard app/*.py app/**/*.py)
APP_DATA := $(wildcard app/data/*)
APP_CONFIG := $(wildcard app/requirements.txt)
SHINYLIVE_DEPS := $(APP_FILES) $(APP_DATA) $(APP_CONFIG)
SHINYLIVE_STAMP := site/.shinylive-built

PYTHON_FILES := $(wildcard app/*.py app/cards/*.py)
TEST_FILES := $(wildcard tests/**/*.py)
TEST_DEPS := $(PYTHON_FILES) $(TEST_FILES)
TEST_STAMP := .make/test-passed

QUARTO_CONFIG := $(wildcard _quarto.yml markdown/_quarto.yml)
QMD_FILES := $(wildcard markdown/*.qmd)
HTML_FILES := $(patsubst markdown/%.qmd,app/www/markdown/%.html,$(QMD_FILES))

preview: $(HTML_FILES)

app/www/markdown/%.html: markdown/%.qmd $(QUARTO_CONFIG)
	quarto render $< --to html --output-dir app/www


.PHONY: preview test test-force clean app shinylive shinylive-force shinylive-serve 

preview: $(HTML_FILES)

test: $(TEST_STAMP)

$(TEST_STAMP): $(TEST_DEPS)
	mkdir -p .make
	python -m pytest
	touch $(TEST_STAMP)

test-force:
	python -m pytest

app:
	$(SHINY) run --reload --launch-browser app/app.py

# Export only when application files have changed.
shinylive: $(SHINYLIVE_STAMP)

$(SHINYLIVE_STAMP): $(SHINYLIVE_DEPS)
	$(SHINYLIVE) export app site
	touch $(SHINYLIVE_STAMP)

# Export if necessary, then serve the static Shinylive version.
shinylive-serve: shinylive
	$(PYTHON) -m http.server 8000 --directory site

# Export regardless of file timestamps.
shinylive-force:
	$(SHINYLIVE) export app site
	touch $(SHINYLIVE_STAMP)



clean:
	rm -f app/www/markdown/*.html
	rm -f $(TEST_STAMP)
	rm -f $(SHINYLIVE_STAMP)
