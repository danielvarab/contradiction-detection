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
PATH_TO_SNLI=$1
PATH_TO_EMBEDDING=$2
EMBEDDING_DIMENSION=$3
GPU_ID=$4
FIX_EMBEDDINGS=$5

# Variables
currentDirectory=`pwd`
tmp=$(basename "$PATH_TO_EMBEDDING")
EMBEDDINGS=${tmp%.*}
OUTPUT_FOLDER=${EMBEDDINGS}"-RUNNING"

if [ ! -d $OUTPUT_FOLDER ]; then
	  # create output directory
	  mkdir ${OUTPUT_FOLDER}
elif [[ -d "$OUTPUT_FOLDER" ]]; then
		# Found a matching folder name, quitting. 
		echo "A folder named ${OUTPUT_FOLDER} does already exists. Quitting to avoid overriding previously generated results."
		exit 1;
fi

# Close STDOUT file descriptor
exec 1<&-
# Close STDERR FD
exec 2<&-

# Open STDOUT as $LOG_FILE file for read and write.
exec 1<>${OUTPUT_FOLDER}/${EMBEDDINGS}.log.txt

# Redirect STDERR to STDOUT
exec 2>&1

echo $@
date +$'\n'"%R:%D BASH INFO:"$'\t'"USING ${EMBEDDINGS} AS INPUT EMBEDDINGS"
date +$'\n'"%R:%D BASH INFO:"$'\t'"OUTPUTTING ALL FILES TO ${OUTPUT_FOLDER}"


# Splitting data
date +$'\n'"%R:%D BASH INFO:"$'\t'"SPLITTING DATA"
python process-snli.py --data_folder $PATH_TO_SNLI --out_folder $OUTPUT_FOLDER

# Preproccess data
date +$'\n'"%R:%D BASH INFO:"$'\t'"PREPROCESSING DATA STEP 1/2"
python preprocess.py \
--srcfile ${OUTPUT_FOLDER}/"src-train.txt" \
--targetfile ${OUTPUT_FOLDER}/"targ-train.txt" \
--labelfile ${OUTPUT_FOLDER}/"label-train.txt" \
--srcvalfile ${OUTPUT_FOLDER}/"src-dev.txt" \
--targetvalfile ${OUTPUT_FOLDER}/"targ-dev.txt" \
--labelvalfile ${OUTPUT_FOLDER}/"label-dev.txt" \
--srctestfile ${OUTPUT_FOLDER}/"src-test.txt" \
--targettestfile ${OUTPUT_FOLDER}/"targ-test.txt" \
--labeltestfile ${OUTPUT_FOLDER}/"label-test.txt" \
--outputfile ${OUTPUT_FOLDER}"/entail" \
--glove $PATH_TO_EMBEDDING \

date +$'\n'"%R:%D BASH INFO:"$'\t'"PREPROCESSING DATA STEP 2/2"
python get_pretrain_vecs.py \
--dictionary ${OUTPUT_FOLDER}"/entail.word.dict" \
--glove $PATH_TO_EMBEDDING \
--outputfile ${OUTPUT_FOLDER}"/glove.hdf5" \
--d $EMBEDDING_DIMENSION

# Training
date +$'\n'"%R:%D BASH INFO:"$'\t'"STARTED TRAINING WITH $OUTPUT_FOLDER"
th train.lua \
-data_file ${OUTPUT_FOLDER}"/entail-train.hdf5" \
-val_data_file ${OUTPUT_FOLDER}"/entail-val.hdf5" \
-test_data_file ${OUTPUT_FOLDER}"/entail-test.hdf5" \
-pre_word_vecs ${OUTPUT_FOLDER}"/glove.hdf5" \
-gpuid $GPU_ID \
-savefile ${OUTPUT_FOLDER}/result.model \
-word_vec_size $EMBEDDING_DIMENSION \
-fix_word_vecs $FIX_EMBEDDINGS

date +$'\n'"%R:%D BASH INFO:"$'\t'"DONE TRAINING WITH $OUTPUT_FOLDER"


# Predict 
date +$'\n'"%R:%D BASH INFO:"$'\t'"STARTED PREDICTING WITH ${OUTPUT_FOLDER}/result.model_final.t7"
th predict.lua \
-sent1_file ${OUTPUT_FOLDER}/"src-test.txt" \
-sent2_file ${OUTPUT_FOLDER}/"targ-test.txt" \
-model ${OUTPUT_FOLDER}/result.model_final.t7 \
-word_dict ${OUTPUT_FOLDER}"/entail.word.dict" \
-label_dict ${OUTPUT_FOLDER}"/entail.label.dict" \
-output_file ${OUTPUT_FOLDER}"/pred.txt"


date +$'\n'"%R:%D BASH INFO:"$'\t'"DONE PREDICTING WITH ${OUTPUT_FOLDER}/result.model_final.t7"


# Confusion Matrix
date +$'\n'"%R:%D BASH INFO:"$'\t'"BUILDING CONFUSION MATRIX"
python confusion.py \
--labels ${OUTPUT_FOLDER}/"label-test.txt" \
--predict ${OUTPUT_FOLDER}"/pred.txt" \
--outfile ${OUTPUT_FOLDER}"/confusion_matrix.txt"

date +$'\n'"%R:%D BASH INFO:"$'\t'"DONE BUILDING CONFUSION MATRIX"

# Cleaning
date +$'\n'"%R:%D BASH INFO:"$'\t'"Cleaning up..."
#snli
rm -v ${OUTPUT_FOLDER}/"src-train.txt"
rm -v ${OUTPUT_FOLDER}/"targ-train.txt"
rm -v ${OUTPUT_FOLDER}/"label-train.txt"
rm -v ${OUTPUT_FOLDER}/"src-dev.txt"
rm -v ${OUTPUT_FOLDER}/"targ-dev.txt"
rm -v ${OUTPUT_FOLDER}/"label-dev.txt"

#train
rm -v ${OUTPUT_FOLDER}"/entail-train.hdf5"
rm -v ${OUTPUT_FOLDER}"/result.model.t7"

if [ -d $OUTPUT_FOLDER ]; then
	  # Control will enter here if $preprocess_directory doesn't exist.
	  mv ${OUTPUT_FOLDER} ${EMBEDDINGS}
fi

date +$'\n'"%R:%D BASH INFO:"$'\t'"COMPLETELY DONE!"