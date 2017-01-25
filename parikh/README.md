# Running Parikhs Generic Sentence Classification implementation

The implementation of this paper [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933). Parikh et al. EMNLP 2016.
Credits goes to <a href="http://yoon.io">Yoon Kim</a> for the implementation of 

## Resources

### Code
https://github.com/harvardnlp/decomp-attn


### Data
Stanford Natural Language Inference (SNLI) dataset can be downloaded from http://nlp.stanford.edu/projects/snli/

Pre-trained GloVe embeddings can be downloaded from http://nlp.stanford.edu/projects/glove/ - Use 300d vectors since the parihk implementation uses 300d vectors as default


## Environment setup
In order to run the training step, Lua and appropriate Torch libraries is needed. 
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh

source ~/.profile

luarocks install hdf5

brew tap homebrew/science
brew install hdf5
```
Update config file accordingly:
you should configure config.lua at Users/your_name/torch/install/share/lua/5.1/hdf5.
Replace the HDF5_INCLUDE_PATH = "/usr/local/Cellar/hdf5/your_version_number/include".
Replace your_name and your_version_number based on your settings.

According to this link: https://github.com/karpathy/neuraltalk2/issues/123


## Splitting the data
```
python split_snli_parikh.py --devfile path-to-dev-file 
                            --trainfile path-to-train-file 
                            --testfile path-to-test-file 
                            --output path-to-output-folder
```

This will create a total of 9 .txt files in the output-folder. 3 files per data input file, corresponding to sent1, sent2 and label, which will be used in the next step.
## Preprocessing

First run:
```
python preprocess.py --srcfile path-to-sent1-train --targetfile path-to-sent2-train
--labelfile path-to-label-train --srcvalfile path-to-sent1-val --targetvalfile path-to-sent2-val
--labelvalfile path-to-label-val --srctestfile path-to-sent1-test --targettestfile path-to-sent2-test
--labeltestfile path-to-label-test --outputfile data/entail --glove path-to-glove
```
This will create the data hdf5 files. Vocabulary is based on the pretrained Glove embeddings,
with `path-to-glove` being the path to the pretrained Glove word vecs (i.e. the `glove.840B.300d.txt`
file)

For natural language inference sent1 can be the premise and sent2 can be the hypothesis.

Now run:
```
python get_pretrain_vecs.py --glove path-to-glove --outputfile data/glove.hdf5
--dictionary path-to-dict
```
`path-to-dict` is the `*.word.dict` file created from running `preprocess.py`



## Training

To train the model, run 
```
th train.lua -data_file path-to-train -val_data_file path-to-val -test_data_file path-to-test
-pre_word_vecs path-to-word-vecs
```
Here `path-to-word-vecs` is the hdf5 file created from running `get_pretrain_vecs.py`.

You can add `-gpuid 1` to use the (first) GPU.

The model essentially replicates the results of Parikh et al. (2016). The main difference is that
they use asynchronous updates, while this code uses synchronous updates.

## Predicting
To predict on new data, run
```
th predict.lua -sent1_file path-to-sent1 -sent2_file path-to-sent2 -model path-to-model
-word_dict path-to-word-dict -label_dict path-to-label-dict -output_file pred.txt
```
This will output the predictions to `pred.txt`. `path-to-word-dict` and `path-to-label-dict` are the
*.dict files created from running `preprocess.py`
