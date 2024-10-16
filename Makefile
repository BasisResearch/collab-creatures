lint: FORCE
	./scripts/lint.sh

format:
	./scripts/clean.sh

test_modules: FORCE
	./scripts/test_modules.sh

test_notebooks: FORCE
	./scripts/test_notebooks.sh

FORCE: