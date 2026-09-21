PYTHON := .venv/bin/python
SHINY := .venv/bin/shiny
SHINYLIVE := .venv/bin/shinylive

# Local modules must precede similarly named third-party packages.
export PYTHONPATH := $(CURDIR)/app$(if $(PYTHONPATH),:$(PYTHONPATH))

APP_FILES := $(wildcard app/*.py app/**/*.py)
APP_DATA := $(wildcard app/data/*)
APP_CONFIG := $(wildcard app/requirements.txt)
SHINYLIVE_DEPS := $(APP_FILES) $(APP_DATA) $(APP_CONFIG)
SHINYLIVE_STAMP := site/.shinylive-built

PYTHON_FILES := $(wildcard app/*.py app/cards/*.py)
TEST_FILES := $(wildcard tests/**/*.py)
TEST_DEPS := $(PYTHON_FILES) $(TEST_FILES) Makefile pyproject.toml pytest.ini
TEST_STAMP := .make/test-passed

QUARTO_CONFIG := $(wildcard _quarto.yml markdown/_quarto.yml)
QMD_FILES := $(wildcard markdown/*.qmd)
HTML_FILES := $(patsubst markdown/%.qmd,app/www/markdown/%.html,$(QMD_FILES))

preview: $(HTML_FILES) app/www/README.html

README.html: README.md $(QUARTO_CONFIG) app/www/pythagoras.css app/www/tetractys.png
	quarto render $< --to html

app/www/README.html: README.html Makefile
	mkdir -p $(@D)
	sed 's|href="app/www/|href="|g' $< > $@

app/www/markdown/%.html: markdown/%.qmd $(QUARTO_CONFIG)
	quarto render $< --to html --output-dir app/www


.PHONY: preview install check-imports test test-force clean app shinylive shinylive-force shinylive-serve

# Refresh the editable source link in the existing environment. Dependencies
# are left alone except for legacy PyPI packages that shadow local modules.
# Ordinary Python edits need only a process restart, not another install.
install:
	$(PYTHON) -m pip install --no-deps --editable .
	$(PYTHON) -m pip uninstall --yes card cards roles
	$(MAKE) check-imports

# Diagnose which files a fresh process will load under these Make targets.
check-imports:
	$(PYTHON) -c 'import importlib.util, pathlib, sys; root = pathlib.Path("app").resolve(); expected = {"card": root / "card.py", "module": root / "module.py", "cards": root / "cards/__init__.py", "roles": root / "roles.py"}; actual = {name: getattr(importlib.util.find_spec(name), "origin", None) for name in expected}; print("Python:", sys.executable); [print(name + ":", path) for name, path in actual.items()]; assert all(actual[name] and pathlib.Path(actual[name]).resolve() == path for name, path in expected.items()), "Import paths do not point to this checkout"'

test: check-imports $(TEST_STAMP)

$(TEST_STAMP): $(TEST_DEPS)
	mkdir -p .make
	$(PYTHON) -m pytest
	touch $(TEST_STAMP)

test-force: check-imports
	$(PYTHON) -m pytest

app: check-imports
	$(SHINY) run --reload --launch-browser app/app.py

# Export only when application files have changed.
shinylive: $(SHINYLIVE_STAMP)

$(SHINYLIVE_STAMP): $(SHINYLIVE_DEPS)
	$(SHINYLIVE) export app site
	touch $(SHINYLIVE_STAMP)

# Export if necessary, then serve the static Shinylive version.
shinylive-serve: shinylive
	python3.14 -m http.server 8000 --directory site --tls-cert ~/.local-certs/site.pem --tls-key ~/.local-certs/site-key.pem

# Export regardless of file timestamps.
shinylive-force:
	$(SHINYLIVE) export app site
	touch $(SHINYLIVE_STAMP)

clean:
	rm -f app/www/markdown/*.html
	rm -f app/www/README.html
	rm -f $(TEST_STAMP)
	rm -f $(SHINYLIVE_STAMP)
