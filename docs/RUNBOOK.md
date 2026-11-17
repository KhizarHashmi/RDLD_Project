# Run book

The full run order, with every command, its outputs and its run time, is in [`PIPELINE.md`](PIPELINE.md).

The first step downloads the data and the model checkpoints:

```
python scripts/p0_download_data.py --models
```

After that, `python scripts/p0_prepare_doclaynet.py` builds the validation page set, and
`RDLD_DATA=../RDLD_data_test SPLIT=test python scripts/p0_prepare_doclaynet.py` builds the test page set.
