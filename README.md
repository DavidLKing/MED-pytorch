# morph-seq2seq
My reimplementation of LMU's MED system. Note that this reimplementation is already outdated. I've included the med.yml for reference if anyone's curious, but the underlying library from IBM is no longer supported.

It requires [pytorch](http://pytorch.org) and [pytorch-seq2seq](https://github.com/DavidLKing/pytorch-seq2seq) (this is my mocked up version---temp fix), , so make sure you have those installed. 

We also now have git flow. Develop will rapidly change and frequently break. Master and release
should be more stable.  Just a forewarning though, this is ALL VERY ALPHA STILL. 

Big todos:
- [x] Add word vectors
- [ ] ~~Fully integrate pytorch-seq2seq into source code (there were lots of edits to make things work)~~
- [x] Fix model loading bug... currently doesn't consistently load what's in the config file
- [ ] New develop verison is a bit of a drastic update: update `main.py` to be compatible with new development branch as exemplified in IBM's new `sample.py` 
- [x] Set up eval on the test set
- [ ] Make it so that we can save models with custom names
- [ ] Set up ensembling, possibly as separate script
- [ ] Add Faruqui attention
- [ ] Make sure config file and command line options are in sync
- [x] Allow saving of model outputs (predictions)
- [ ] ~~Clean out `results` folder upon final release~~
- [ ] **MAJOR DEVIATION FROM MED, encode features separately.**

`preprocess.py SIGMORPHON_FILE` generates `data.txt`, `vocab.source`, and `vocab.target`.
`make_splits.py data.txt` generates train, dev, and test splits. 

- `data.txt` --- Is a table separated file with with source on the left and targets on the right.
- `vocab.*` files have exactly one vocab element

`./main.py --config config.yml` Runs the model in *train* or *eval* mode.

Current sigmorphon2016 German dev score: 

`Val accuracy: 1524 out of 1597 0.9542892924232936`

acc on sig set sans adjectives:

`Val accuracy: 609 out of 673 0.9049034175334324`

Russian acc with word vectors:
`Val accuracy: 1453 out of 1591 0.9132620993086109`