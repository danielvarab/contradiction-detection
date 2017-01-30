#!/bin/bash


# Example run
# ./train.sh path-to-snli-data path-to-glove-vectors | tee log.txt

# Installation of Lua + Torch + dependencies
# in a terminal, run the commands WITHOUT sudo
# git clone https://github.com/torch/distro.git ~/torch --recursive
# cd ~/torch; bash install-deps;
# ./install.sh
# source ~/.bashrc
# luarocks install hdf5



# Command line Arguments 
path_to_snli_data=$1
path_to_glove_vectors=$2

# Variables
number_of_sentences=30000
splitted_data=splitted_data/
currentDirectory=`pwd`

# Use FD 19 to capture the debug stream caused by "set -x":
#exec 19>my-script.txt
# Tell bash about it  (there's nothing special about 19, its arbitrary)
#BASH_XTRACEFD=19

# turn on the debug stream:
#set -x

# Splitting data
date +$'\n'"%R:%D BASH INFO:"$'\t'"SPLITTING DATA"
cd $path_to_snli_data
python "$currentDirectory/split_snli_parikh.py" \
--n $number_of_sentences \
--devfile "snli_1.0_dev.txt" \
--trainfile "snli_1.0_dev.txt" \
--testfile "snli_1.0_dev.txt" \
--output $currentDirectory"/"${splitted_data}

# Preproccess
date +$'\n'"%R:%D BASH INFO:"$'\t'"PREPROCESSING DATA"
cd $currentDirectory
DIRECTORY=data
if [ ! -d "$DIRECTORY" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir $DIRECTORY
fi

python preprocess.py \
--srcfile ${splitted_data}"train-sentence1-"${number_of_sentences}"-SNLI.txt" \
--targetfile $splitted_data"train-sentence2-"$number_of_sentences"-SNLI.txt" \
--labelfile $splitted_data"train-gold_label-"$number_of_sentences"-SNLI.txt" \
--srcvalfile $splitted_data"val-sentence1-"$number_of_sentences"-SNLI.txt" \
--targetvalfile $splitted_data"val-sentence2-"$number_of_sentences"-SNLI.txt" \
--labelvalfile $splitted_data"val-gold_label-"$number_of_sentences"-SNLI.txt" \
--srctestfile $splitted_data"dev-sentence1-"$number_of_sentences"-SNLI.txt" \
--targettestfile $splitted_data"dev-sentence1-"$number_of_sentences"-SNLI.txt" \
--labeltestfile $splitted_data"dev-gold_label-"$number_of_sentences"-SNLI.txt" \
--outputfile ${DIRECTORY}"/entail" \
--glove $path_to_glove_vectors

#python get_pretrain_vecs.py \
#--glove $path_to_glove_vectors \
#--outputfile ${DIRECTORY}"/glove.hdf5" \
#--dictionary ${DIRECTORY}"/entail.word.dict" \

# Training
date +$'\n'"%R:%D BASH INFO:"$'\t'"TRAINING"
#th train.lua \
#-data_file ${DIRECTORY}"/entail-train.hdf5" \
#-val_data_file ${DIRECTORY}"/entail-val.hdf5" \
#-test_data_file ${DIRECTORY}"/entail-test.hdf5" \
#-pre_word_vecs ${DIRECTORY}"/glove.hdf5"
