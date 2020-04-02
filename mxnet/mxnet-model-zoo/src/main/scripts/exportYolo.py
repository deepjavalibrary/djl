#!/usr/bin/env python3

import os
import json
import mxnet as mx
from gluoncv import model_zoo

toExport = [
    ("darknet53", "voc", (320, 320)),
    ("darknet53", "voc", (416, 416)),
    ("mobilenet1.0", "voc", (320, 320)),
    ("mobilenet1.0", "voc", (416, 416)),
    ("darknet53", "coco", (320, 320)),
    ("darknet53", "coco", (416, 416)),
    ("darknet53", "coco", (608, 608)),
    ("mobilenet1.0", "coco", (320, 320)),
    ("mobilenet1.0", "coco", (416, 416)),
    ("mobilenet1.0", "coco", (608, 608)),
    ]

version = "0.0.1"

metadata = {
        'metadataVersion': 0.1,
        'groupId': 'ai.djl.mxnet',
        'artifactId': 'yolo',
        'name': 'yolo',
        'description': 'Yolo GluonCV Model',
        'website': 'https://gluon-cv.mxnet.io/',
        'licenses': {
            'license': {
                'name': 'The Apache License, Version 2.0',
                'url': 'https://www.apache.org/licenses/LICENSE-2.0'
                }
            },
        'artifacts': []
        }

for baseModel, dataset, imageSize in toExport:
    mx.autograd.set_training(False)
    mx.autograd.set_recording(False)
    modelName = "yolo3_" + baseModel + "_" + dataset
    print("Exporting", modelName, imageSize)
    height, width = imageSize
    artifact = {
            'version': version,
            'snapshot': False,
            'name': 'yolo',
            'properties': {
                'dataset': dataset,
                'version': '3',
                'backbone': baseModel,
                'imageSize': height
                },
            'arguments': {
                'threshold': 0.2
                },
            'files': dict()
            }
    net = model_zoo.get_model(modelName, pretrained=True)
    imageShape = (32, 3, height, width)
    x = mx.nd.random.normal(shape=imageShape)
    net(x)
    net.hybridize()
    net(x)

    dirName = "out/" + version + "/" + modelName + "-" + str(height) + "x" + str(width) + "/"
    os.makedirs(dirName)
    net.export(dirName + "yolo")

    if dataset == "coco":
        artifact['files']['classes'] = {
                'uri': 'https://mlrepo.djl.ai/model/cv/object_detection/ai/djl/mxnet/classes_coco.txt',
                'sha1Hash': '1febf3c237fb06e472a001fd8e03f16cc6174090',
                'name': 'classes.txt',
                'size': 620
                }
    elif dataset == "voc":
        artifact['files']['classes'] = {
                'uri': 'https://mlrepo.djl.ai/model/cv/object_detection/ai/djl/mxnet/classes_voc.txt',
                'sha1Hash': 'c6796ef8b46238f33366d94f3e16801cbda9e7f8',
                'name': 'classes.txt',
                'size': 134
                }

    symbolFilename = dirName + "yolo-symbol.json"
    artifact['files']['symbol'] = {
            'uri': symbolFilename[4:],
            'sha1Hash': os.popen("gsha1sum " + symbolFilename).read().split()[0],
            'size': os.path.getsize(symbolFilename)
            }

    paramFilename = dirName + "yolo-0000.params"
    os.popen("gzip " + paramFilename).read()
    paramFilename = paramFilename + ".gz"
    artifact['files']['parameters'] = {
            'uri': paramFilename[4:],
            'sha1Hash': os.popen("gsha1sum " + paramFilename).read().split()[0],
            'size': os.path.getsize(paramFilename)
            }

    metadata['artifacts'].append(artifact)

with open('out/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
