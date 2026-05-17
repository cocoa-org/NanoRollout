FROM ghcr.io/bowenbryanwang/uda-desktop:0.6.0

USER root

RUN pip install "weasyprint" "pymupdf" "requests"
RUN apt-get update && apt-get install -y ffmpeg

RUN install -d -m 0755 -o kasm-user -g kasm-user /tmp_workspace /tmp_workspace/results

COPY --chown=kasm-user:kasm-user skills /home/kasm-user/skills

USER 1000
WORKDIR /tmp_workspace
