from mxnet import nd
from gluoncv.model_zoo import get_model
import argparse
import os

parser = argparse.ArgumentParser(description='This is the gluoncv model export portal')
parser.add_argument("--name", required=True, type=str, help="This is the name of the available gluon models")
parser.add_argument("--output_path", required=True, type=str, help="This is the output path of the model")
parser.add_argument("--shape", required=True, type=str, help="This is the input shape of the model, like (1, 3, 224, 224)")

def export(model_name, input, export_path):
    net = get_model(model_name, pretrained=True)
    net.hybridize(static_alloc = True, static_shape = True)
    net(input)
    os.mkdir(export_path)
    net.export(export_path + "/" + model_name)

if __name__ == "__main__":
    args = parser.parse_args()
    input = nd.zeros(eval(args.shape))
    export(args.name, input, args.output_path)
