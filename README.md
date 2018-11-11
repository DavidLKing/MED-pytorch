# morph-seq2seq
My reimplementation of LMU's MED system

It requires [pytorch](http://pytorch.org) and [pytorch-seq2seq](https://github.com/DavidLKing/pytorch-seq2seq) (this is my mocked up version---I found a few bugs, fixed them, and there is a pull request pending), , so make sure you have those installed. 

Big todos:
- [ ] Set up eval on the test set
- [ ] Make it so that we can save models with custom names
- [ ] Set up ensembling, possibly as separate script
- [ ] Add Faruqui attention
- [ ] Make sure config file and command line options are in sync
- [ ] Allow saving of model outputs (predictions)
- [ ] Clean out `results` folder upon final release

`preprocess.py SIGMORPHON_FILE` generates `data.txt`, `vocab.source`, and `vocab.target`.

- `data.txt` --- Is a table separated file with with source on the left and targets on the right.
- `vocab.*` files have exactly one vocab element

`./main.py --config config.yml` Runs the model in *train* or *eval* mode.

Current sigmorphon2016 German dev score: 

`Val accuracy: 1524 out of 1597 0.9542892924232936`

acc on sig set sans adjectives:

`Val accuracy: 609 out of 673 0.9049034175334324`
