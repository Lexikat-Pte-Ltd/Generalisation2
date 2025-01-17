.ONESHELL:

# Debug related
VSCODE = code

# Dataset related 
SOURCE_FOLDER := ./data
PUBLISH_FOLDER := ./data
DATASET_PATTERN := *_*_*_*_*:*_run_data_at_*_*.json

# Docker compose related 
MINI_DIR := ./docker/mini-learn-compose
FEDORA_DIR := ./docker/fedora-learn-compose
CONTAINER ?= mini

# Tmux related
SESSION_NAME := learn-compose

# Set the appropriate directory based on the container type
ifeq ($(CONTAINER),mini)
    DOCKER_DIR := $(MINI_DIR)
    SHELL := /bin/sh
else ifeq ($(CONTAINER),fedora)
    DOCKER_DIR := $(FEDORA_DIR)
    SHELL := /bin/bash
else
    $(error Invalid CONTAINER value. Use 'mini' or 'fedora')
endif

# Main rules
debug-mode: mini-up-container fedora-up-container
full-rund: mini-up-container fedora-up-container run-maind
full-run: mini-up-container fedora-up-container run-main
full-loop: mini-up-container fedora-up-container run-loop

run-main: 
	python scripts/main.py

run-maind: 
	python scripts/main.py --debug

run-loop:
	@if tmux has-session -t $(SESSION_NAME) 2>/dev/null; then \
		echo "Attaching to existing loop session: $(SESSION_NAME)"; \
		tmux attach-session -t $(SESSION_NAME); \
	else \
		echo "Creating new loop session: $(SESSION_NAME)"; \
		tmux new-session -s $(SESSION_NAME) "./scripts/infinite_run.sh"; \
	fi

# Container-specific rules
mini-up-container:
	$(MAKE) CONTAINER=mini up-container

mini-down-container:
	$(MAKE) CONTAINER=mini down-container

mini-restart-container:
	$(MAKE) CONTAINER=mini restart-container

mini-rebuild-container:
	$(MAKE) CONTAINER=mini rebuild-container

mini-attach-container:
	$(MAKE) CONTAINER=mini attach-container

fedora-up-container:
	$(MAKE) CONTAINER=fedora up-container

fedora-down-container:
	$(MAKE) CONTAINER=fedora down-container

fedora-restart-container:
	$(MAKE) CONTAINER=fedora restart-container

fedora-rebuild-container:
	$(MAKE) CONTAINER=fedora rebuild-container

fedora-attach-container:
	$(MAKE) CONTAINER=fedora attach-container

# Dataset 
archive-data:
	./scripts/archive_old_runs.sh ./data/

publish-data:
	@find $(SOURCE_FOLDER) -maxdepth 1 -type f \( -name 'ca*' -o -name 'ea*' \) -exec cp -v {} $(PUBLISH_FOLDER) \;

# Generic Rules
up-container:
	cd $(DOCKER_DIR)
	docker compose up -d

down-container:
	cd $(DOCKER_DIR)
	docker compose down

restart-container: down-container up-container
	cd $(DOCKER_DIR)
	docker compose logs -f

rebuild-container: down-container
	cd $(DOCKER_DIR)
	docker compose up -d --force-recreate --build
	docker compose down

attach-container:
	cd $(DOCKER_DIR) && \
	SERVICE_NAME=$$(docker compose ps --services | head -n 1) && \
	if [ -n "$$SERVICE_NAME" ]; then \
		docker compose exec -it $$SERVICE_NAME $(SHELL); \
	else \
		echo "No running service found"; \
	fi

# Test related
test:
	python -m pytest test/

test-verbose:
	python -m pytest -v test/

test-coverage:
	python -m pytest --cov=src test/

test-watch:
	ptw -- test/
