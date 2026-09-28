FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install uv
RUN uv sync

RUN uv run python -m spacy download en_core_web_md

CMD ["uv", "run", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]