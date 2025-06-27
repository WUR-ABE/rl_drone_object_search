ARG BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04

###########################################
FROM ${BASE_IMAGE} as base

ENV DEBIAN_FRONTEND=noninteractive

# Install language
RUN apt-get update && apt-get install -y \
  locales \
  && locale-gen en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*
ENV LANG en_US.UTF-8

# Install timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y upgrade \
  && rm -rf /var/lib/apt/lists/*

# Install common programs
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  gnupg2 \
  lsb-release \
  sudo \
  software-properties-common \
  wget \
  && rm -rf /var/lib/apt/lists/*

################
# Expose the nvidia driver to allow opengl 
# Dependencies for glvnd and X11.
################
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
  libglvnd0 \
  libgl1 \
  libglx0 \
  libegl1 \
  libxext6 \
  libx11-6

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
# enable all capabilities for the container
# Explained here: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.10.0/user-guide.html#driver-capabilities
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV QT_X11_NO_MITSHM 1

###########################################
#  Develop image
###########################################
FROM base AS dev

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  bash-completion \
  build-essential \
  cmake \
  gdb \
  git \
  git-lfs \
  openssh-client \
  python3-argcomplete \
  python3-pip \
  vim \
  && rm -rf /var/lib/apt/lists/*

ARG USERNAME=abe
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Delete the root user https://askubuntu.com/questions/1513927/ubuntu-24-04-docker-images-now-includes-user-ubuntu-with-uid-gid-1000
RUN userdel -r ubuntu

# Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  # Add sudo support for the non-root user
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*

# Set up autocompletion for user
RUN apt-get update && apt-get install -y git-core bash-completion \
  && rm -rf /var/lib/apt/lists/* 

ENV DEBIAN_FRONTEND=

###########################################
#  Full image
###########################################
FROM dev AS full

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=

###########################################
FROM dev as abe

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -y lsb-release curl software-properties-common apt-transport-https && \
  apt-get clean

ENV DEBIAN_FRONTEND=

###########################################
FROM abe as openglvnc

ARG ENTRY_POINT=/opt/entrypoint.sh
ENV ENTRY_POINT=${ENTRY_POINT}

ENV DEBIAN_FRONTEND=noninteractive
RUN curl -L -O https://github.com/VirtualGL/virtualgl/releases/download/3.1.1/virtualgl_3.1.1_amd64.deb && \
  apt-get -y install ./virtualgl_3.1.1_amd64.deb && \
  rm virtualgl_3.1.1_amd64.deb
RUN curl -L -O https://github.com/TurboVNC/turbovnc/releases/download/3.1.1/turbovnc_3.1.1_amd64.deb && \
  apt-get -y install ./turbovnc_3.1.1_amd64.deb && \
  rm turbovnc_3.1.1_amd64.deb
RUN addgroup --gid 1002 vglusers && adduser abe video && adduser abe vglusers
RUN apt-get update && apt-get -y install xfce4-session xfce4-panel xfce4-terminal thunar xterm x11-utils python3-minimal python3-pip python3-numpy unzip less tmux screen
ENV DEBIAN_FRONTEND=

# Install noVNC

ENV NOVNC_VERSION=1.4.0
ENV WEBSOCKETIFY_VERSION=0.10.0

RUN mkdir -p /usr/local/novnc && \
  curl -sSL https://github.com/novnc/noVNC/archive/v${NOVNC_VERSION}.zip -o /tmp/novnc-install.zip && \
  unzip /tmp/novnc-install.zip -d /usr/local/novnc && \
  cp /usr/local/novnc/noVNC-${NOVNC_VERSION}/vnc.html /usr/local/novnc/noVNC-${NOVNC_VERSION}/index.html && \
  curl -sSL https://github.com/novnc/websockify/archive/v${WEBSOCKETIFY_VERSION}.zip -o /tmp/websockify-install.zip && \
  unzip /tmp/websockify-install.zip -d /usr/local/novnc && \
  ln -s /usr/local/novnc/websockify-${WEBSOCKETIFY_VERSION} /usr/local/novnc/noVNC-${NOVNC_VERSION}/utils/websockify && \
  rm -f /tmp/websockify-install.zip /tmp/novnc-install.zip && \
  sed -i -E 's/^python /python3 /' /usr/local/novnc/websockify-${WEBSOCKETIFY_VERSION}/run

RUN cat <<EOF > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
    {
        "file_format_version" : "1.0.0",
        "ICD" : {
            "library_path" : "libEGL_nvidia.so.0"
        }
    }
EOF

COPY .devcontainer/start-turbovnc.sh /opt/nvidia/entrypoint.d/90-turbovnc.sh
COPY .devcontainer/start-turbovnc.sh /opt/entrypoint.d/90-turbovnc.sh

COPY .devcontainer/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
EXPOSE 5801

###########################################
FROM openglvnc as user
USER abe
ENV HOME=/home/abe
WORKDIR ${HOME}
RUN sudo apt-get purge -y xfce4-screensaver

ENV DISPLAY=:1
ENV TVNC_VGL=1
ENV VGL_ISACTIVE=1
ENV VGL_FPS=30
ENV VGL_COMPRESS=0
ENV VGL_DISPLAY=egl
ENV VGL_WM=1
ENV VGL_PROBEGLX=0
ENV LD_PRELOAD=libdlfaker.so:libvglfaker.so
ENV PIP_BREAK_SYSTEM_PACKAGES=1
