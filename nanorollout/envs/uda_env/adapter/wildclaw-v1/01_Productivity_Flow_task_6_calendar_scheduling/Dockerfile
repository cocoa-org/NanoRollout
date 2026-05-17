FROM ghcr.io/bowenbryanwang/uda-desktop:0.6.0

USER root

RUN pip install --no-cache-dir "ortools>=9.0" "PyYAML>=6.0"

RUN install -d -m 0755 -o kasm-user -g kasm-user /tmp_workspace /tmp_workspace/results

USER 1000
WORKDIR /tmp_workspace
