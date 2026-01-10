run:
	python main.py

run-dev:
	python main.py --stage dev

run-dashboard:
	python main.py --stage dashboard

docker-build:
	docker build -t atsuvovor/cyberthreat-insight .

docker-run:
	docker run --rm atsuvovor/cyberthreat-insight

docker-push:
	docker push atsuvovor/cyberthreat-insight:latest
