.ONESHELL: 
.PHONY: seaweed seaweed-up seaweed-down seaweed-restart seaweed-attach

run-main: up-test-container
	python scripts/main.py

up-test-container:
	cd docker/learn-compose/
	docker compose up -d

down-test-container:
	cd docker/learn-compose/
	docker compose down 

restart-test-container: down-test-container up-test-container
	cd docker/learn-compose/
	docker compose logs -f

rebuild-test-container: down-test-container 
	cd docker/learn-compose/
	docker compose up -d --build
	docker compose down

