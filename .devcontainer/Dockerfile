FROM python:3.11-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic system packages
RUN apt-get update && apt-get install -y \
    git \
    curl \
    bash \
    bash-completion \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Optional: Install only if necessary for your Python dependencies
# RUN apt-get update && apt-get install -y build-essential

# Copy and install Python dependencies from .devcontainer
COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set an optional working directory
WORKDIR /workspace

# Add a basic .bashrc for better shell experience
RUN echo '\n\
# Enable tab completion\n\
if [ -f /etc/bash_completion ] && ! shopt -oq posix; then\n\
    . /etc/bash_completion\n\
fi\n\
# Enable history navigation with arrow keys\n\
bind '\"\\e[A\":history-search-backward'\n\
bind '\"\\e[B\":history-search-forward'\n\
# Increase history size\n\
export HISTSIZE=10000\n\
export HISTFILESIZE=10000\n\
' > /home/$USERNAME/.bashrc && chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc

USER $USERNAME
