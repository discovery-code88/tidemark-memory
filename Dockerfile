# Tidemark: build the service in a Rust image, run it in a slim Debian image.
# Both stages use Debian trixie: the prebuilt ONNX Runtime linked in for the optional reranker needs
# glibc >= 2.38 and a GCC 13+ C++ runtime (bookworm has 2.36 and GCC 12), and run time must match build time.
FROM rust:1.93-trixie AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked

FROM debian:trixie-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --home /data tidemark \
 && mkdir -p /data && chown tidemark /data
COPY --from=build /src/target/release/tidemark /usr/local/bin/tidemark
USER tidemark
ENV TIDEMARK_HOST=0.0.0.0 \
    TIDEMARK_PORT=18200 \
    TIDEMARK_DB=/data/tidemark.sqlite \
    MALLOC_ARENA_MAX=2
VOLUME /data
EXPOSE 18200
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD ["/bin/bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/18200 && printf 'GET /health HTTP/1.0\\r\\n\\r\\n' >&3 && grep -q '\"ok\"' <&3"]
CMD ["tidemark"]
