COMPOSE = docker compose -f docker-compose.yml
ENV_FILE = .env

up:
	$(COMPOSE) --env-file $(ENV_FILE) up -d

down:
	$(COMPOSE) --env-file $(ENV_FILE) down

logs:
	$(COMPOSE) --env-file $(ENV_FILE) logs -f

ps:
	$(COMPOSE) --env-file $(ENV_FILE) ps

pull:
	$(COMPOSE) --env-file $(ENV_FILE) pull

restart:
	$(COMPOSE) --env-file $(ENV_FILE) restart

health:
	@for c in postgres redis qdrant vllm dcgm-exporter n8n open-webui prometheus grafana loki promtail node-exporter alertmanager; do \
	  status=$$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}(no healthcheck){{end}}' $$c 2>/dev/null || echo "not running"); \
	  printf "%-20s %s\n" "$$c" "$$status"; \
	done

generate-secrets:
	@echo "POSTGRES_PASSWORD=$$(openssl rand -base64 32)"
	@echo "REDIS_PASSWORD=$$(openssl rand -base64 32)"
	@echo "N8N_ENCRYPTION_KEY=$$(openssl rand -hex 24)"
	@echo "N8N_JWT_SECRET=$$(openssl rand -hex 24)"
	@echo "WEBUI_SECRET_KEY=$$(openssl rand -hex 32)"
	@echo "GRAFANA_ADMIN_PASSWORD=$$(openssl rand -base64 24)"

backup-postgres:
	mkdir -p backups/postgres
	$(COMPOSE) --env-file $(ENV_FILE) exec -T postgres \
	  pg_dumpall -U $$(grep ^POSTGRES_USER $(ENV_FILE) | cut -d= -f2) \
	  | gzip > backups/postgres/$$(date +%Y%m%d_%H%M%S).sql.gz
	@echo "Backup saved to backups/postgres/"

restore-postgres:
	@test -n "$(FILE)" || (echo "Usage: make restore-postgres FILE=backups/postgres/<name>.sql.gz" && exit 1)
	gunzip -c $(FILE) | $(COMPOSE) --env-file $(ENV_FILE) exec -T postgres \
	  psql -U $$(grep ^POSTGRES_USER $(ENV_FILE) | cut -d= -f2) postgres

backup-qdrant:
	@test -n "$(COLLECTION)" || (echo "Usage: make backup-qdrant COLLECTION=<name>" && exit 1)
	mkdir -p backups/qdrant
	$(COMPOSE) --env-file $(ENV_FILE) exec -T qdrant \
	  sh -c 'curl -sf -X POST http://localhost:6333/collections/$(COLLECTION)/snapshots' \
	  | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4 \
	  | xargs -I {} sh -c ' \
	    echo "Snapshot: {}" && \
	    curl -sf "http://localhost:6333/collections/$(COLLECTION)/snapshots/{}" \
	      -o "backups/qdrant/{}" && \
	    echo "Downloaded to backups/qdrant/{}"'

export-images:
	mkdir -p backups/images
	$(COMPOSE) --env-file $(ENV_FILE) config --images | sort -u \
	  | xargs docker save \
	  | gzip > backups/images/ai-platform-images-$$(date +%Y%m%d).tar.gz

import-images:
	@test -n "$(BUNDLE)" || (echo "Usage: make import-images BUNDLE=<path/to/images.tar.gz>" && exit 1)
	gunzip -c $(BUNDLE) | docker load

.PHONY: up down logs ps pull restart health generate-secrets backup-postgres restore-postgres backup-qdrant export-images import-images
