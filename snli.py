import pandas as pd


# basic script to convert snli parquet into csv


# train dataset
df = pd.read_parquet("data/raw/snli-train.parquet")
df.drop(df[df.label==-1].index, inplace=True)
df.to_csv("data/snli-train.csv")

# dev dataset
df = pd.read_parquet("data/raw/snli-dev.parquet")
df.drop(df[df.label==-1].index, inplace=True)
df.to_csv("data/snli-dev.csv")

# test dataset
df = pd.read_parquet("data/raw/snli-test.parquet")
df.drop(df[df.label==-1].index, inplace=True)
df.to_csv("data/snli-test.csv")

# after converting to csv, manually update first column header to 'id'