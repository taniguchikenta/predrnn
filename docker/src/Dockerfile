ARG PYTORCH_VERSION="1.9.0"
ARG CUDA_VERSION="11.1"
ARG CUDNN_VERSION="8"
ARG IMAGE_FLAVOR="devel"

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-${IMAGE_FLAVOR}
# パッケージの取得元をlinux.yz.yamagata-u.ac.jpに変更
RUN perl -p -i.bak -e 's%(deb(?:-src|)\s+)https?://(?!archive\.canonical\.com|security\.ubuntu\.com)[^\s]+%$1http://linux.yz.yamagata-u.ac.jp/pub/linux/ubuntu/archives/%' /etc/apt/sources.list

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

LABEL maintainer="TaniguchiKenta"

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    nano \
    gedit \
    tar \
    python3.6


RUN apt-get autoremove -y

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libglib2.0-0 &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
    rich \
    opencv-contrib-python==4.7.0.68 \
    tensorflow \
    matplotlib \
    scikit-learn \
    scikit-image \
    lpips

# userの設定
ARG username=predrnn
ARG wkdir=/home/${username}/
ARG UID=1000
ARG GID=1000
ARG HOSTNAME
RUN adduser --uid ${UID} --disabled-password --gecos '' --shell /bin/bash ${username}&& \
    chown -R ${username}:${username} /home/${username} && \
    echo "%${username}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${username} && \
    chmod 0440 /etc/sudoers.d/${username}
# workdirの権限をルートからユーザーに変更
# RUN chown ${username}:${username} ${wkdir}
# bashrcの設定
# ADD ./.bashrc /home/${username}/
# RUN chmod +x /home/${username}/.bashrc
# ADD ./init.sh /
# RUN chmod +x /init.sh
# RUN echo "source /home/predrnn/.bashrc" >> /root/.bash_profile
ADD ./.bashrc /home/${username}/
RUN chmod +x /home/${username}/.bashrc

RUN echo "export PS1='\[\033[01;32m\]${username}@${HOSTNAME}\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /home/${username}/.bashrc && \
    echo 'alias python3="/usr/bin/python3.6"' >> /home/${username}/.bashrc && \
    echo "PYTHONIOENCODING=utf-8" >> /home/${username}/.bashrc && \
    echo 'alias pip="pip3"' >> /home/${username}/.bashrc



# ユーザー
USER ${username}
WORKDIR /home/${username}/shared_dir

# CMD ["/bin/bash", "-c", "source .bashrc"]
# CMD ["source /home/predrnn/.bashrc", "-c"]

# ENTRYPOINT ["/init.sh"]
# CMD ["/bin/bash", "--rcfile", "/.bashrc"]
# CMD ["/bin/bash", "--rcfile", "/home/predrnn/.bashrc"]
# CMD [".bashrc", "-l"]
# ENTRYPOINT ["/home/predrnn/.bashrc", "-l"]
# CMD ["/bin/bash", "-l", "-c", ". /.bashrc"]
