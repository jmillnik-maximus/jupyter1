FROM ubuntu:latest

### INSTALL BLOCK ###

RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN apt-get install -y curl

RUN pip3 -q install pip --upgrade

COPY . .
# RUN pip3 install -r requirements.txt

RUN pip3 install tensorflow
RUN pip3 install matplotlib
RUN pip3 install scikit-image

 RUN pip3 install jupyter
 WORKDIR /src/notebooks

# RUN pip3 install ipython

# CMD ["/usr/bin/python3", "/src/hello.py"]
# CMD ["/usr/local/bin/ipython", "/src/module.py"]

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
