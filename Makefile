.ONESHELL: 

run-main: up-test-container
	python scripts/main.py

run-main-d: up-test-container
	python scripts/main.py --debug

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

attach-test-container:
	cd docker/learn-compose/ && \
	SERVICE_NAME=$$(docker compose ps --services | head -n 1) && \
	if [ -n "$$SERVICE_NAME" ]; then \
		docker compose exec -it $$SERVICE_NAME /bin/sh; \
	else \
		echo "No running service found"; \
	fi

