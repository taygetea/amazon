### preliminary exploratory analysis of Amazon.com review data using word2vec

We are relying on [datasets provided by Julian McAuley](http://jmcauley.ucsd.edu/data/amazon/).

Given Python 3 and Pip, install the dependencies (may require root privileges for a global installation) with:

```
pip3 install -r requirements.txt
```

(Some of the dependencies may also be available through your distribution's package manager; for example, while preparing this README, the present author used the `python3-scipy` package from the Ubuntu Trusty 14.04 LTS repositories.)

Download and unpack the review data. The 3.1 GiB gzipball is 8.9 GiB unpacked, so be sure to do this somewhere you have a lot of space, unlike a default EC2 EBS volume!

```
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
gunzip reviews_Books_5.json.gz
```
