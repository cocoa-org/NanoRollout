FROM ghcr.io/bowenbryanwang/uda-desktop:0.6.0

USER root

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install "openai>=1.0.0"

RUN install -d -m 0755 -o kasm-user -g kasm-user /tmp_workspace /tmp_workspace/results
COPY --chown=kasm-user:kasm-user skills /home/kasm-user/skills

USER 1000
WORKDIR /tmp_workspace
