SHELL := /bin/bash

# Detect how to open things depending on our OS
OS = $(shell uname -s)
ifeq ($(OS),Linux)
	OPEN=xdg-open
else
	OPEN=open
endif

PROFILE = default
# Import env variables
include .env.shared
$(shell touch .env)
include .env

# Allow us to execute make commands from within our project's conda env
# TODO: add over-ride based on some environment variable?
#       e.g. `MAKE_NO_ENV` in `.env` set makes this do nothing
define execute_in_env
	source bin/conda_activate.sh && conda_activate && $1
endef

.PHONY: init
## Fully initialise a project: install; setup github repo; setup S3 bucket
init: install setup-github setup-bucket
	@echo INIT COMPLETE

.PHONY: install
## Install a project: create conda env; install local package; setup git hooks; setup metaflow+AWS
install: conda-create setup-git setup-metaflow
	@echo INSTALL COMPLETE

.PHONY: inputs-pull
## Pull `inputs/` from S3
inputs-pull:
	$(call execute_in_env, aws s3 sync s3://${BUCKET}/inputs inputs --profile ${PROFILE})

.PHONY: inputs-push
## Push `inputs/` to S3 (WARNING: this may overwrite existing files!)
inputs-push:
	$(call execute_in_env, aws s3 sync inputs s3://${BUCKET}/inputs  --profile ${PROFILE})

.PHONY: docs
## Build the API documentation
docs:
	$(call execute_in_env, sphinx-apidoc -o docs/api ${REPO_NAME})
	$(call execute_in_env, sphinx-build -b docs/ docs/_build)

.PHONY: docs-clean
## Clean the built API documentation
docs-clean:
	rm -r docs/source/api
	rm -r docs/_build

.PHONY: docs-open
## Open the docs in the browser
docs-open:
	$(OPEN) docs/_build/index.html

.PHONY: conda-create
## Create a conda environment
conda-create:
	conda env create -q -n ${PROJECT_NAME} -f environment.yaml
	$(MAKE) -s pip-install

.PHONY: conda-update
## Update the conda-environment based on changes to `environment.yaml`
conda-update:
	conda env update -n ${PROJECT_NAME} -f environment.yaml
	$(MAKE) pip-install

.PHONY: clean
## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: pre-commit
## Perform pre-commit actions
pre-commit:
	$(call execute_in_env, pre-commit)

.PHONY: lint
## Run flake8 linting on repository
lint:
	$(call execute_in_env, flake8)

.PHONY: pip-install
## Install our package and requirements in editable mode (including development dependencies)
pip-install:
	$(call execute_in_env, pip install -e ".[dev]" --quiet)

#################################################################################
# Helper Commands (no need to explicitly document)                              #
#################################################################################

.PHONY: setup-git
setup-git:
	$(call execute_in_env, pre-commit install --install-hooks)

.PHONY: setup-metaflow
setup-metaflow:
	$(call execute_in_env, ${SHELL} ./bin/install_metaflow_aws.sh)

.PHONY: setup-bucket
setup-bucket:
	@echo S£
	$(call execute_in_env, ${SHELL} ./bin/create_bucket.sh)

.PHONY: setup-github
setup-github:
	@echo GH
	$(call execute_in_env, ${SHELL} ./bin/create_repo.sh)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
