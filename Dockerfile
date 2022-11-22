FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ARG USERNAME=konakona
ARG USER_UID=1000
ARG USER_GID=888


# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt install software-properties-common -y \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
WORKDIR /home/konakona

ENV PATH="/home/konakona/miniconda3/bin:${PATH}"
ARG PATH="/home/konakona/miniconda3/bin:${PATH}"


RUN sudo apt install -y build-essential
RUN sudo apt-get install -y wget
RUN sudo apt -y install git
RUN sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt-get install python3.8 -y
RUN sudo ln -s /usr/bin/pip3 /usr/bin/pip && \
    sudo ln -s /usr/bin/python3.8 /usr/bin/python
