FROM mariadb:10.8
RUN apt-get update && apt-get install -y bash openssl openssh-server openssh-client python3 python3-pip python3-venv python3-dev gcc linux-headers-generic procps coreutils util-linux && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash alice
ENV VIRTUAL_ENV=/home/alice/venv
RUN python3 -m venv /home/alice/venv
ENV PATH=/home/alice/venv/bin:$PATH
ENV PYTHONPATH=
RUN ln -sf /usr/bin/python3 /usr/local/bin/python
RUN ln -sf /home/alice/venv/bin/python3 /usr/local/bin/python3
RUN ln -sf /home/alice/venv/bin/pip /usr/local/bin/pip
RUN ln -sf /home/alice/venv/bin/pip3 /usr/local/bin/pip3
ADD --chown=alice:alice documents.tar.gz /home/alice/
ADD --chown=alice:alice tmp.tar.gz /home/alice/
ADD --chown=alice:alice var.tar.gz /home/alice/
COPY --chown=root:root randomly_encrypt.sh /home/alice/
COPY --chown=root:root set_random_permissions.sh /home/alice/
COPY --chown=alice:alice .ssh /home/alice/.ssh/
COPY --chown=alice:alice requirements.txt /home/alice/
RUN chmod +x /home/alice/randomly_encrypt.sh
RUN chmod +x /home/alice/set_random_permissions.sh
RUN cd /home/alice && ./randomly_encrypt.sh
RUN cd /home/alice && ./set_random_permissions.sh
RUN chmod 700 /home/alice/.ssh
RUN . /home/alice/venv/bin/activate && pip install --no-cache-dir -r /home/alice/requirements.txt
WORKDIR /home/alice
USER alice
RUN echo "source /home/alice/venv/bin/activate" >> ~/.bashrc
ENTRYPOINT ["bash"]
CMD ["-l"]