import pandas
import random

filename = "./SG568_1-2/fold_tr2.csv"

n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
s = int(n/2) #desired sample size
print("\nDB: {}\n\nTotal #: {}\tselected: {}".format(filename,n,s))
skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pandas.read_csv(filename, skiprows=skip)

df.to_csv(filename, index=False)