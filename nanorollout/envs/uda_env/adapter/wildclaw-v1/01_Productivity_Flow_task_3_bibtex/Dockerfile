FROM ghcr.io/bowenbryanwang/uda-desktop:0.6.0

USER root

RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils && rm -rf /var/lib/apt/lists/*

RUN install -d -m 0755 -o kasm-user -g kasm-user /tmp_workspace /tmp_workspace/results

USER 1000
WORKDIR /tmp_workspace
