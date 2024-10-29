lint: FORCE
	./scripts/lint.sh

format:
	./scripts/clean.sh

test_modules: FORCE
	./scripts/test_modules.sh

test_notebooks: FORCE
	./scripts/test_notebooks.sh

gendoc: FORCE
	cd docs && rm -rf _build && make html

docserve: FORCE
	python -m http.server -d docs/_build/html 8080

FORCE: