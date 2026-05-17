FROM ghcr.io/bowenbryanwang/uda-desktop:0.6.0

USER root

RUN pip install "openai" "playwright"
RUN python3 -m playwright install chromium
RUN install -d -m 0755 -o kasm-user -g kasm-user /tmp_workspace /tmp_workspace/results

USER 1000
WORKDIR /tmp_workspace
