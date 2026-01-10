# =========================================
# CyberThreat_Insight Makefile
# =========================================

# Default stage to run
STAGE ?= all
IMAGE_NAME = atsuvovor/cyberthreat-insight
STREAMLIT_PORT = 8501

# =========================================
# Local pipeline execution
# =========================================
run:
	@echo "▶ Running full CyberThreat_Insight pipeline"
	python main.py --stage all

run-stage:
	@echo "▶ Running pipeline stage: $(STAGE)"
	python main.py --stage $(STAGE)

run-dashboard:
	@echo "▶ Running Streamlit dashboard"
	streamlit run app.py

# =========================================
# Docker commands
# =========================================
docker-build:
	@echo "▶ Building Docker image: $(IMAGE_NAME)"
	docker build -t $(IMAGE_NAME) .

docker-run:
	@echo "▶ Running Docker container (stage: $(STAGE))"
	docker run --rm $(IMAGE_NAME) --stage $(STAGE)

docker-run-dashboard:
	@echo "▶ Running Streamlit dashboard in Docker"
	docker run -p $(STREAMLIT_PORT):$(STREAMLIT_PORT) $(IMAGE_NAME) streamlit run app.py

docker-push:
	@echo "▶ Pushing Docker image to Docker Hub: $(IMAGE_NAME)"
	docker push $(IMAGE_NAME)

# =========================================
# Clean outputs (optional)
# =========================================
clean:
	@echo "▶ Cleaning outputs, logs, and temp files"
	rm -rf outputs/* logs/*
