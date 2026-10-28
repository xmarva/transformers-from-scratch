# syntax=docker/dockerfile:1.28.0@sha256:bb22d9815c728170f72750f4e5b0d672e06176142e1d602c7e66c050100b7e5b
# perevod's server with its model inside: the same commit builds the same image, bit for bit.
FROM python:3.12-slim@sha256:12316debbfb86890f6b2d825820612741ba527ad5a44e3902759141d572bfe8f

# a fixed time instead of the clock: with it pip writes .pyc files that don't record when
ARG SOURCE_DATE_EPOCH
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN useradd --uid 1000 --create-home perevod
WORKDIR /home/perevod

# the dependencies first, exactly as locked: they change least often, so this layer is cached
COPY requirements.lock build.lock ./
RUN pip install --require-hashes --no-deps -r requirements.lock -r build.lock \
        --extra-index-url https://download.pytorch.org/whl/cpu

# the package itself, built with the locked build backend
COPY pyproject.toml README.md ./src/
COPY src ./src/src
RUN pip install --no-deps --no-build-isolation ./src && rm -rf ./src

# the model, at the commit this version of perevod was tested with; the downloader's log is
# named by the time of the download, so it goes
USER perevod
RUN python -c "from perevod import Translator; Translator.from_pretrained()" \
    && rm -rf /home/perevod/.cache/huggingface/xet
ENV HF_HUB_OFFLINE=1

ARG VERSION=dev REVISION=unknown
LABEL org.opencontainers.image.source=https://github.com/xmarva/transformers-from-scratch \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$REVISION
EXPOSE 8000
CMD ["perevod-server", "--host", "0.0.0.0", "--port", "8000"]
