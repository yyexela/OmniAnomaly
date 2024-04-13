# Base image
FROM nvcr.io/nvidia/tensorflow:23.03-tf1-py3

# Resolves error with key
# See: https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# See: https://askubuntu.com/questions/1444943/nvidia-gpg-error-the-following-signatures-couldnt-be-verified-because-the-publi
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Update image contents to have latest python3 and pip3 for image
RUN apt-get update
#RUN apt install -y --no-install-recommends python3-dev python-is-python3
# python3-setuptools
# python3-pip 
RUN apt remove -y python3-pip
RUN python3 -m pip uninstall -y pip 
#RUN python -m easy_install pip

#RUN python3 -m pip install --upgrade pip
WORKDIR /usr/local/bin
RUN ln -s /usr/bin/python3 python
#RUN pip3 install --upgrade pip
#RUN apt install -y python3-pip --reinstall
RUN apt-get install -y git curl zip unzip tmux

# Create /app directory
WORKDIR /app

# Copy OmniAnomaly requirements into image
COPY ./requirements.txt /app

# Install OmniAnomaly requirements
RUN pip3 install -r requirements.txt

# Set initial folder to be OmniAnomaly
WORKDIR /app/OmniAnomaly
