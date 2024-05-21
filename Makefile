# Makefile

.PHONY: seaweed seaweed-up seaweed-down seaweed-restart seaweed-attach

seaweed:
	@echo "Starting SeaweedFS..."
	docker-compose -f docker/dataset-centre-seaweedfs/docker-compose.yml up -d

seaweed-down:
	@echo "Stopping SeaweedFS..."
	docker-compose -f docker/dataset-centre-seaweedfs/docker-compose.yml down

seaweed-restart: 
	seaweed-down seaweed

seaweed-logs:
	@echo "Logging to 'gen2-seaweedfs'..."
	docker logs gen2-seaweedfs

learn:
	@echo "Starting learning container..."
	docker-compose -f docker/learn-env/docker-compose.yml up

learn-down:
	@echo "Stopping learning container..."
	docker-compose -f docker/learn-env/docker-compose.yml down

learn-restart: 
	seaweed-down seaweed


