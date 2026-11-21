FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Clone sqlglot and find the exact commit containing blob fed01b2 (from sqlglot/dialects/dialect.py)
RUN git clone https://github.com/tobymao/sqlglot.git /app/repo \
    && cd /app/repo \
    && TARGET_COMMIT="" \
    && for c in $(git log --format="%H" -n 800 sqlglot/dialects/dialect.py); do \
         BLOB=$(git rev-parse "$c:sqlglot/dialects/dialect.py" 2>/dev/null || true); \
         case "$BLOB" in fed01b2*) TARGET_COMMIT="$c"; break ;; esac; \
       done \
    && if [ -n "$TARGET_COMMIT" ]; then git checkout "$TARGET_COMMIT"; fi \
    && pip install --no-cache-dir --no-deps -e /app/repo

COPY build_map.py /app/build_map.py
COPY selector.py /app/selector.py

RUN python3 /app/build_map.py /app/repo /app/coverage_map.json \
    && rm -rf /app/repo/.git

CMD ["python", "/app/selector.py"]