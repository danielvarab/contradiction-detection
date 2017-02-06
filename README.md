# contradiction-detection


## Word Embeddings

All embeddings have a overview at http://wordvectors.org/suite.php

### SkipGram (Word2Vec)
#### How to download and translate to txt

1. Download the word2vec binary embedding from  https://code.google.com/archive/p/word2vec/
2. Clone the following repo https://github.com/marekrei/convertvec and follow its readme
3. Running this will produce a .txt file with a header line (this is not a valid format)
4. To translate into a valid format run the following command
`tail -n +2 infile.txt > outfile.txt` (note this will generate the file twice. to avoid a temporary file run this instead `echo "$(tail -n +2 file.txt)" > file.txt`)

### GloVe

- May be downloaded from http://nlp.stanford.edu/projects/glove/ (exists in different versions [6B](http://nlp.stanford.edu/data/glove.6B.zip), [42B](http://nlp.stanford.edu/data/glove.6B.zip), [840B](http://nlp.stanford.edu/data/glove.840B.300d.zip))

### Global Context Vectors

- [direct link](http://nlp.stanford.edu/~socherr/
ACL2012_wordVectorsTextFile.zip)

### Multilingual Vectors

1. Downloadable via the authors site wordvectors.org ([direct link](http://www.wordvectors.org/web-eacl14-vectors/de-projected-en-512.txt.gz))
2. This is a zip including 2 files, vocab.txt and wordVectors.txt. To produce a combined file, run the following file: `paste -d " " vocab.txt wordVectors.txt > multi-vectors.txt`
