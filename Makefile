.PHONY: format
format:
	isort .
	autoflake --remove-all-unused-imports -i -r --exclude __init__.py .
	black .