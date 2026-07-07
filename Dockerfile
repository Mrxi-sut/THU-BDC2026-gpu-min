FROM python:3.12-slim-bookworm AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    libc6-dev \
    make \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Build the TA-Lib C runtime once, then copy only runtime files into final image.
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

COPY requirements-gpu.txt /tmp/requirements-gpu.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 && \
    pip install --no-cache-dir -r /tmp/requirements-gpu.txt && \
    find /opt/venv -type d \( -name "__pycache__" -o -name "tests" -o -name "test" \) -prune -exec rm -rf '{}' + && \
    find /opt/venv -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

RUN rm -rf /opt/venv/lib/python3.12/site-packages/triton \
           /opt/venv/lib/python3.12/site-packages/triton-*.dist-info \
           /opt/venv/lib/python3.12/site-packages/torch/include && \
    find /opt/venv/lib/python3.12/site-packages/nvidia -type d -name include -prune -exec rm -rf '{}' +


FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /usr/lib/libta_lib.* /usr/lib/
COPY --from=deps /usr/include/ta-lib /usr/include/ta-lib
COPY --from=deps /opt/venv /opt/venv

COPY code ./code
COPY model ./model
COPY init.sh train.sh test.sh ./
COPY README.md ./readme.md

RUN mkdir -p data output temp && ldconfig

ENV PATH="/opt/venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

CMD ["sleep", "infinity"]
