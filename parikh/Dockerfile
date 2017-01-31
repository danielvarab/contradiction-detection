# Usage: FROM [image name]
FROM ubuntu


# Usage: RUN [command]
RUN apt-get -qq update
RUN apt-get -qq upgrade --yes
RUN apt-get -y -qq install sudo
RUN apt-get -qq install git --yes
RUN apt-get -qq install wget
RUN sudo apt-get -qq install python-pip python-dev build-essential --yes
RUN pip install --upgrade pip


# install torch
RUN git clone https://github.com/torch/distro.git ~/torch --recursive
RUN cd ~/torch; bash install-deps;
RUN cd ~/torch; ./install.sh;

# hdf5 
RUN sudo apt-get -qq install libhdf5-serial-dev hdf5-tools --yes
RUN ~/torch/install/bin/luarocks install hdf5


# Getting code
RUN cd ~/
RUN git clone https://github.com/danielvarab/contradiction-detection
ADD contradiction-detection
RUN mkdir contradiction-detection/datasets
RUN cd contradiction-detection/parikh
RUN pip install -r contradiction-detection/parikh/requirements.txt

# Getting datasets
RUN wget -q http://nlp.stanford.edu/projects/snli/snli_1.0.zip -o ~/contradiction-detection/datasets/snli_1.0.zip
RUN unzip ~/contradiction-detection/datasets/snli_1.0.zip
RUN mkdir ~/contradiction-detection/datasets/glove
RUN wget -q http://nlp.stanford.edu/data/glove.6B.zip ~/contradiction-detection/datasets/glove/glove.6B.zip
RUN unzip ~/contradiction-detection/datasets/glove/glove.6B.zip




