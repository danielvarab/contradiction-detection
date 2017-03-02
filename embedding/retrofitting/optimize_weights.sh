# Command line Arguments 
PATH_TO_EMBEDDING=$1

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


date +$'\n'"%R:%D BASH INFO:"$'\t'"USING ${EMBEDDINGS} AS INPUT EMBEDDINGS"
date +$'\n'"%R:%D BASH INFO:"$'\t'"OUTPUTTING ALL FILES TO ${OUTPUT_FOLDER}"

# Declare list for grid run
declare -a arr=(1 2 5 10 20 50)

# Start retrofitting
date +$'\n'"%R:%D BASH INFO:"$'\t'"RUNNING RETROFIT TO OPTIMIZE BETA AND GAMMA"
for i in ${arr[@]}
do
	for j in ${arr[@]}
	do
		python retrofit.py \
		--e $PATH_TO_EMBEDDING \
		--beta $i \
		--gamma $j \
		--normalize \
		--a_s_rf \
		--outfolder $OUTPUT_FOLDER \
	done
done

#Evaluate all retrofittings
date +$'\n'"%R:%D BASH INFO:"$'\t'"EVALUATING ALL NEW RETROFITS"

python ../evaluation/evaluate_embedding.py \
--d $OUTPUT_FOLDER \
--ws \
--ss \
--sa \
--dc \
--de \
> optimization_results.out


if [ -d $OUTPUT_FOLDER ]; then
	  # Control will enter here if $preprocess_directory doesn't exist.
	  mv ${OUTPUT_FOLDER} ${EMBEDDINGS}
fi
