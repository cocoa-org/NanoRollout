FROM ghcr.io/bowenbryanwang/uda-desktop:0.6.0

USER root

RUN pip install -q fastapi uvicorn

RUN install -d -m 0755 -o kasm-user -g kasm-user /tmp_workspace /tmp_workspace/results

COPY --chown=kasm-user:kasm-user skills /home/kasm-user/skills

USER 1000
WORKDIR /tmp_workspace
