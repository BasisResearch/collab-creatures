lint: FORCE
	./scripts/lint.sh

format:
	./scripts/clean.sh

tests: FORCE
	python -m pytest -v tests
	./scripts/test_notebooks.sh

FORCE: