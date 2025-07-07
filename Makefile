.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: run
run:
    docker run -p 3000:3000 alfaarghya/alfa-leetcode-api:2.0.1
	python main.py
