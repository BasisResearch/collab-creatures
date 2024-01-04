lint: FORCE
	./scripts/lint.sh

format:
	./scripts/clean.sh

test: FORCE
	pytest -v tests
	./scripts/test_notebooks.sh

FORCE: