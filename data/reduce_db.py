import pandas
import random


red = 32

basedb = "./SG568_1-" + str(red) + "/"


for fold in ["fold_val0.csv", "fold_val1.csv", "fold_val2.csv"]:

    filename = basedb + fold

    # number of records in file (excludes header)
    n = sum(1 for line in open(filename)) - 1
    s = int(n / red)  # desired sample size
    print("\nDB: {}\n\nTotal #: {}\tselected: {}".format(filename, n, s))
    # the 0-indexed header will not be included in the skip list
    skip = sorted(random.sample(range(1, n + 1), n - s))
    df = pandas.read_csv(filename, skiprows=skip)

    df.to_csv(filename, index=False)
