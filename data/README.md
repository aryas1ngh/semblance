# Data

The **Stanford Online Products (SOP)** dataset is **not committed** (≈3 GB zip,
≈6 GB extracted, 120k images). Fetch it locally:

```bash
curl -L -C - -o data/Stanford_Online_Products.zip \
  "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"
unzip -q data/Stanford_Online_Products.zip -d data/
```

This yields `data/Stanford_Online_Products/` containing `Ebay_train.txt`,
`Ebay_test.txt`, and the 12 `*_final/` category image folders.

**Citation.** Song, H. O., Xiang, Y., Jegelka, S., & Savarese, S. (2016).
*Deep Metric Learning via Lifted Structured Feature Embedding.* CVPR.

The standard split is used unchanged: 11,318 train classes / 11,316 test
classes, with **zero class overlap** (open-set retrieval).
