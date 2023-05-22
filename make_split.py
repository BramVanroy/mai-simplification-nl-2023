from pathlib import Path
from typing import Optional

import pandas as pd


def main(
        fin: str,
        train_p: float,
        dev_p: float,
        test_p: float,
        sep: str = "\t",
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None
):
    """Given an input file, split it into train, dev, test splits according to the given percentages
    :param fin: input data file
    :param train_p: percentage training data
    :param dev_p: percentage dev data
    :param test_p: percentage test data
    :param sep: separator to use to read the input file (output seperator will always be a comma)
    :param shuffle: whether to shuffle the data before splitting
    :param shuffle_seed: optional seed to use for deterministic shuffling
    """
    df = pd.read_csv(fin, encoding="utf-8", sep=sep)

    if train_p + dev_p + test_p != 1.:
        raise ValueError(f"'train_p', 'dev_p' and 'test_p' must sum to 1!")

    # Shuffle dataset before splitting. If 'shuffle_seed' is not None, shuffling is deterministic
    if shuffle:
        df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)

    # Calculate the sizes of the splits based on the percentages
    total_rows = len(df)
    train_size = int(total_rows * train_p)
    dev_size = int(total_rows * dev_p)

    # Split the DataFrame into train, dev, and test dfs
    train_data = df[:train_size]
    dev_data = df[train_size:train_size + dev_size]
    test_data = df[train_size + dev_size:]
    print(f"TOTAL SIZE: {total_rows:,}")
    print(f"TRAIN SIZE: {len(train_data):,}; DEV SIZE: {len(dev_data):,}; TEST SIZE: {len(test_data):,}")

    pdout = Path(fin).parent.joinpath("splits")
    pdout.mkdir(exist_ok=True)

    train_data.to_csv(pdout / "train.csv", index=False)
    dev_data.to_csv(pdout / "validation.csv", index=False)
    test_data.to_csv(pdout / "test.csv", index=False)


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Split a given dataset into train, dev, test splits")
    cparser.add_argument("fin", help="Input CSV data file")
    cparser.add_argument("train_p", type=float, help="Size of training set in percentage")
    cparser.add_argument("dev_p", type=float, help="Size of dev set in percentage")
    cparser.add_argument("test_p", type=float, help="Size of test set in percentage")
    cparser.add_argument("--sep", default="\t", help="Separator in the CSV file (defaults to tab character)")
    cparser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the dataset before splitting")
    cparser.add_argument("--shuffle_seed", type=int,
                         help="Shuffle seed to fix. If given, the shuffle will be deterministic")

    main(**vars(cparser.parse_args()))
