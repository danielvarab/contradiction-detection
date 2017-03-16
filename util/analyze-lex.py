# antonyms.txt
#  - neighbours: 380447
#  - stddev: 30.63
#  - mean: 15.43

# synonyms.txt
#  - neighbours: 1011466
#  - stddev: 32.65
#  - mean: 18.007

# wordnet.txt
#  - neighbours: 311696
#  - stddev: 3.279
#  - mean: 2.11

# wordnet+.txt
#  - neighbours: 960135
#  - stddev: 17.12
#  - mean: 6.517


def analyze(dic):
  neighbours = list(dic.values())
  counts = [len(elem) for elem in neighbours]
  return np.std(counts), np.mean(counts), sum(counts)

def read_lexicon(filename):
  lexicon = {}
  for line in open(filename, 'r'):
    words = line.lower().strip().split()
    lexicon[words[0] = [word for word in words[1:]]
  return lexicon
