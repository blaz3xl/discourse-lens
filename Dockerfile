# Native installation remains the primary researcher experience.  This image is
# an equivalent isolated runtime for reproducible CLI/MCP operation and publishes
# no network ports.
# Keep the OS/Python base and dependency resolver explicit; exact Python packages
# come from uv.lock rather than floating index resolution during image builds.
# Keep the build resolver immutable; update the version tag and index digest together.
FROM ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 AS uv
FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/discourselens/.venv/bin:$PATH"

WORKDIR /opt/discourselens
ARG DISCOURSELENS_INSTALL_OCR=0
ARG DISCOURSELENS_INSTALL_NLP=0
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra corpus --extra vectors --extra mcp --extra web --no-install-project
COPY src ./src
COPY docs/methodology ./docs/methodology
COPY schemas ./schemas
COPY scripts ./scripts
RUN if [ "$DISCOURSELENS_INSTALL_NLP" = "1" ]; then \
      uv sync --locked --no-dev --extra corpus --extra vectors --extra mcp --extra web --extra nlp --no-editable; \
    else \
      uv sync --locked --no-dev --extra corpus --extra vectors --extra mcp --extra web --no-editable; \
    fi \
    && useradd --create-home --uid 10001 discourselens

# Tesseract and its language data are an opt-in OS-level capability. The base
# image remains smaller and reports OCR as unavailable unless this build arg is 1.
RUN if [ "$DISCOURSELENS_INSTALL_OCR" = "1" ]; then \
      apt-get update \
      && apt-get install --yes --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa tesseract-ocr-nld \
      && rm -rf /var/lib/apt/lists/*; \
    fi

USER 10001:10001
WORKDIR /workspace
CMD ["discourselens", "--help"]
