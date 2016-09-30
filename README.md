# Span-Based Constituency Parser

#### Required Dependencies

 * Python 2.7
 * NumPy
 * [PyCNN](https://github.com/clab/cnn/blob/master/INSTALL.md)

#### Vocabulary Files

Vocabulary may be loaded every time from a training tree file, or it may be stored (separately from a trained model) in a JSON, which is much faster and recommended. To learn the vocabulary from a file with training trees and write a JSON file, use a command such as the following:

```
python src/main.py --train data/02-21.10way.clean --write-vocab data/vocab.json
```

#### Training

Training requires a file containing training trees (`--train`) and a file containg validation trees (`--dev`), which are parsed four times per training epoch to determine which model to keep. A file name must also be provided to store the saved model (`--model'). The following is an example of a command to train a model with all of the default settings:

```
python src/main.py --train data/02-21.10way.clean --dev data/22.auto.clean --vocab data/vocab.json --model data/my_model
```

