FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY paperdigest ./paperdigest
RUN pip install --no-cache-dir .

VOLUME /vault

ENTRYPOINT ["paperdigest"]
CMD ["--help"]
