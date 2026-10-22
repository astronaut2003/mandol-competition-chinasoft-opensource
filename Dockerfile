FROM node:24-bookworm-slim AS studio
WORKDIR /build
COPY studio/package.json studio/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY studio/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS backend
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_CACHE_DIR=/tmp/uv-cache
WORKDIR /app
# Isolated image build: no system package manager, private checkout or model weights.
RUN pip install --no-cache-dir uv==0.10.9
COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY packages/substrate/ packages/substrate/
COPY src/ src/
RUN uv sync --locked --no-dev --no-editable && rm -rf /tmp/uv-cache
COPY demo/ demo/
COPY --from=studio /build/dist/ studio/dist/
RUN groupadd --gid 10001 mandol && useradd --uid 10001 --gid mandol --no-create-home mandol && mkdir /app/data && chown mandol:mandol /app/data
USER mandol
ENV PATH=/app/.venv/bin:$PATH CHINASOFT_APP_ROOT=/app
EXPOSE 20600
CMD ["uvicorn", "mandol_competition.chinasoft.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "20600", "--workers", "1", "--no-access-log", "--timeout-graceful-shutdown", "90"]
