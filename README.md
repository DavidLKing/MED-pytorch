# morph-seq2seq
My reimplementation of LMU's MED system

Big todos:
- [ ] Set up eval on the test set
- [ ] Make it so that we can save models with custom names
- [ ] Set up ensembling, possibly as separate script
- [ ] Add Faruqui attention
- [ ] Make sure config file and command line options are in sync
- [ ] Allow saving of model outputs (predictions)

`preprocess.py SIGMORPHON_FILE` generates `data.txt`, `vocab.source`, and `vocab.target`.

- `data.txt` --- Is a table separated file with with source on the left and targets on the right.
- `vocab.*` files have exactly one vocab element

`./main.py --config config.yml` Runs the model in *train* or *eval* mode.

Current sigmorphon2016 German dev score: 

`Val accuracy: 1524 out of 1597 0.9542892924232936`
