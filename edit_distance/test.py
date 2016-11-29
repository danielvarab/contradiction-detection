import pandas as pd

file_path = "/home/danielvarab/school/contradiction-detection/sick_dataset/SICK_train.txt"
df = pd.read_csv(file_path, sep="\t")

print(df)
