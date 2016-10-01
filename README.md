# Span-Based Constituency Parser

#### Required Dependencies

 * Python 2.7
 * NumPy
 * [PyCNN](https://github.com/clab/cnn/blob/master/INSTALL.md)

#### Vocabulary Files

Vocabulary may be loaded every time from a training tree file, or it may be stored (separately from a trained model) in a JSON file, which is much faster and recommended. To learn the vocabulary from a file with training trees and write a JSON file, use a command such as the following:

```
python src/main.py --train data/02-21.10way.clean --write-vocab data/vocab.json
```

#### Training

Training requires a file containing training trees (`--train`) and a file containg validation trees (`--dev`), which are parsed four times per training epoch to determine which model to keep. A file name must also be provided to store the saved model (`--model`). The following is an example of a command to train a model with all of the default settings:

```
python src/main.py --train data/02-21.10way.clean --dev data/22.auto.clean --vocab data/vocab.json --model data/my_model
```

The following table provides an overview of additional training options:

Argument | Description | Default
--- | --- | ---
--word-dims | Word embedding dimensions | 50
--tag-dims  | POS embedding dimensions  | 20
--lstm-units | LSTM units (per direction, for each of 2 layers) | 200
--hidden-units | Units for ReLU FC layer (each of 2 action types) | 200
--epochs | Number of training epochs | 10
--batch-size | Number of sentences per training update | 10
--droprate | Dropout probability | 0.5
--unk-param | Parameter z for random UNKing | 0.8375
--alpha | Softmax weight for exploration | 1.0
--beta | Oracle action override probability | 0.0


#### Test Evaluation

There is also a facility to directly evaluate a model agaist a reference corpus, by supplying the `--test` argument:

```
python src/main.py --train data/02-21.10way.clean --test data/23.auto.clean --vocab data/vocab.json --model data/my_model
```