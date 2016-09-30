# Span-Based Constituency Parser

#### Required Dependencies

 * Python 2.7
 * NumPy
 * [PyCNN](https://github.com/clab/cnn/blob/master/INSTALL.md)

#### Usage

Vocabulary may be loaded upon both training and application from a file with training trees, or it may be stored (separately from a trained model) in a JSON. To learn the vocabulary from a file with training trees and write a JSON file, use a command such as the following:

```
python src/main.py --train data/02-21.10way.clean --write-vocab data/vocab.json
```