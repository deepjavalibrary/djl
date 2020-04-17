# Model Uploader

## Introduction
This module contains several classes to process artifacts.
You can convert your model from Python using our toolkits and build `metadata.json` to use locally or upload to model zoo.

## Running instruction

### GluonCV model import
You need to make sure all requirement defined in `python/mxnet/requirement.txt` has been met for your python.
You can also create a virtual environment by doing these:
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirement.txt
```
After that, you can execute:
```
./gradlew run --args="-gluoncv -ai 'resnet' -an 'resnet18' -s '(1,3,224,224)' -py '<your-path-to-venv>/venv/bin/python'"
```
This will download pretrained resnet18 model from GluonCV and import to DJL model zoo.
