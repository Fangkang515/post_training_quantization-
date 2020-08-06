import os

import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models
import torchvision.transforms as transforms
import torch.quantization
from torch.quantization import get_default_qconfig, quantize_jit
from utils import accuracy, calibrate, evaluate, print_size_of_model


def get_parser():
    parser = argparse.ArgumentParser(description = 'perform post training quantization on models implemented in torchvision')
    parser.add_argument(
        '-m', '--model', type=str, metavar='',
                        help='selecting the models implemented in torchvision. \n 1.resnet18 \n 2.alexnet 3.squeezenet \n 4.vgg16 \n 5.densenet \n 6.inception \n 7.googlenet \n 8.shufflenet \n 9.mobilenet \n 10.resnext50_32x4d \n11.wide_resnet50_2 \n12.mnasnet')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='', help='saved model checkpoint path')
    parser.add_argument('-d', '--data', type=str, metavar='', help='data path')
    parser.add_argument('-n', '--target', type=int, metavar='', help='number of classes')
    return parser 


def data_loaders(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'valid')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_batch_size = 30
    eval_batch_size = 30

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler)

    return data_loader, data_loader_test


def model_setup(args):
    model_path = args.checkpoint
    
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.model == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif args.model == 'squeezenet':
        model = models.squeezenet1_0(pretrained=True)
    elif args.model == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.model == 'densenet':
        model = models.densenet161(pretrained=True)
    elif args.model == 'inception':
        model = models.inception_v3(pretrained=True)
    elif args.model == 'googlenet':
        model = models.googlenet(pretrained=True)
    elif args.model == 'shufflenet':
        model = models.shufflenet_v2_x1_0(pretrained=True)
    elif args.model == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
    elif args.model == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
    elif args.model == 'wide_resnet50_2':
        model = models.wide_resnet50_2(pretrained=True)
    elif args.model == 'mnasnet':
        model = models.mnasnet1_0(pretrained=True)
    else:
        raise ValueError('please enter a proper model name')
        #model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.target)    
    device = torch.device("cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model


def quantize(model, data_loader, data_loader_test):
    criterion = nn.CrossEntropyLoss()
    num_calibration_batches = 10
    num_eval_batches = 10

    eval_batch_size = 30

    myModel = model
    myModel.eval()
    myModel.qconfig = torch.quantization.default_qconfig
    print(myModel.qconfig)
    torch.quantization.prepare(myModel, inplace=True)

    ts_model = torch.jit.script(model).eval()
    qconfig = get_default_qconfig('fbgemm')
    qconfig_dict = {'': qconfig}
    quantized_model = quantize_jit(
    ts_model,
    {'': qconfig},
    calibrate,
    [data_loader_test])

    print(quantized_model.graph)

    print('Size of model before quantization')
    print_size_of_model(ts_model)
    print('Size of model after quantization')
    print_size_of_model(quantized_model)
    top1, top5 = evaluate(ts_model, criterion, data_loader_test, num_eval_batches)
    print('[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f'%(top1.avg, top5.avg))

    saved_model_dir = '/ssd3/jhahn/ptq/model/saved_model/'
    graph_mode_model_file = 'resnet18_graph_mode_quantized.pth'
    torch.jit.save(quantized_model, saved_model_dir + graph_mode_model_file)
    quantized_model = torch.jit.load(saved_model_dir + graph_mode_model_file)
    top1, top5 = evaluate(quantized_model, criterion, data_loader_test, num_eval_batches)
    print('[after serialization/deserialization] Evaluation accuracy on test dataset: %2.2f, %2.2f'%(top1.avg, top5.avg))


def main():
    #initializing data loader 
    args = get_parser().parse_args()
    #data_path = '/ssd3/jhahn/ptq/data/imagenet_1k/'
    #data_loader, data_loader_test = data_loaders(data_path)
    #model = model_setup()
    model = model_setup(args)
    data_loader, data_loader_test = data_loaders(args)
    quantize(model, data_loader, data_loader_test)


if __name__ == '__main__':
  main()
    
