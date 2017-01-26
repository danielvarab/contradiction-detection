import sys
import pandas as pd
arg_length = len(sys.argv)

if(arg_length < 2):
    print(">> needs path argument")
    sys.exit()

in_path = sys.argv[1]
df = pd.read_csv(in_path, sep="\t") # "../../snli_1.0/snli_1.0_dev.txt"
pair_df = df[["sentence1", "sentence2"]]

if(arg_length > 2):
    out_path = sys.argv[2]
    pair_df.to_csv(out_path, sep="\t", index=False, header=False) # "../../fast_align/build/input.txt"

# then run sed to replace the tabs
# sed -i -e 's/\t/ ||| /g' input.txt

# then run the fast_align program
# ./fast_align -i input.txt -d -o -v > output.txt
