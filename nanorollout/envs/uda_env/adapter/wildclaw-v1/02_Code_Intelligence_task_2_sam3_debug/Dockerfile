FROM ghcr.io/bowenbryanwang/uda-desktop:0.6.0

USER root

RUN pip install --no-cache-dir "numpy==1.26.4" "opencv-python==4.9.0.80"

RUN install -d -m 0755 -o kasm-user -g kasm-user /tmp_workspace /tmp_workspace/results

USER 1000
WORKDIR /tmp_workspace
