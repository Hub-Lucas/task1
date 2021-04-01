import argparse
import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
import os


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=-1,
                        help='embedding_dim')
    parser.add_argument('--max_seq_len', type=int, default=60,
                        help='max_seq_len')

    parser.add_argument('--config', type=str, default="no_file_exists",
                        help='gpu number')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')

    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='learning_rate')

    parser.add_argument('--model', type=str, default="",
                        help='model name')

    parser.add_argument('--position', type=bool, default=False,
                        help='gpu number')

    parser.add_argument('--keep_dropout', type=float, default=0.8,
                        help='keep_dropout')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu number')
    parser.add_argument('--gpu_num', type=int, default=1,
                        help='gpu number')

    #
    args = parser.parse_args()

    if args.config != "no_file_exists":
        if os.path.exists(args.config):
            config = configparser.ConfigParser()
            config_file_path = args.config
            config.read(config_file_path)
            config_common = config['COMMON']
            for key in config_common.keys():
                args.__dict__[key] = config_common[key]
        else:
            print("config file named %s does not exist" % args.config)

    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.model == "transformer":
        args.position = True
    else:
        args.position = False

    # process the type for bool and list
    for arg in args.__dict__.keys():
        if type(args.__dict__[arg]) == str:
            if args.__dict__[arg].lower() == "true":
                args.__dict__[arg] = True
            elif args.__dict__[arg].lower() == "false":
                args.__dict__[arg] = False
            elif "," in args.__dict__[arg]:
                args.__dict__[arg] = [int(i) for i in args.__dict__[arg].split(",")]
            else:
                pass

    return args


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.model_name = 'BaseModel'
        self.opt = opt
        self.fc = nn.Linear(opt.embedding_dim, opt.label_size)

        self.properties = {"model_name": self.__class__.__name__,
                           "batch_size": self.opt.batch_size,
                           "learning_rate": self.opt.learning_rate,
                           "keep_dropout": self.opt.keep_dropout,
                           }

    def forward(self, content):
        # content_ = t.mean(self.encoder(content), dim=1)
        out = self.fc(content_.view(content_.size(0), -1))
        return out

    def save(self, save_dir="saved_model", metric=None):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.model_info = "__".join(
            [k + "_" + str(v) if type(v) != list else k + "_" + str(v)[1:-1].replace(",", "_").replace(",", "") for k, v
             in self.properties.items()])
        if metric:
            path = os.path.join(save_dir, str(metric)[2:] + "_" + self.model_info)
        else:
            path = os.path.join(save_dir, self.model_info)
        t.save(self, path)
        return path


class Inception(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True):
        super(Inception, self).__init__()
        assert (co % 4 == 0)
        cos = [int(co / 4)] * 4
        self.activa = nn.Sequential()
        if norm: self.activa.add_module('norm', nn.BatchNorm1d(co))
        if relu: self.activa.add_module('relu', nn.ReLU(True))
        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[0], 1, stride=1)),
        ]))
        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[1], 1)),
            ('norm1', nn.BatchNorm1d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[1], cos[1], 3, stride=1, padding=1)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[2], 3, padding=1)),
            ('norm1', nn.BatchNorm1d(cos[2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[2], cos[2], 5, stride=1, padding=2)),
        ]))
        self.branch4 = nn.Sequential(OrderedDict([
            # ('pool',nn.MaxPool1d(2)),
            ('conv3', nn.Conv1d(cin, cos[3], 3, stride=1, padding=1)),
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = self.activa(torch.cat((branch1, branch2, branch3, branch4), 1))
        return result


class InceptionCNN(BaseModel):
    def __init__(self, opt):
        super(InceptionCNN, self).__init__(opt)
        incept_dim = getattr(opt, "inception_dim", 512)
        self.model_name = 'CNNText_inception'
        # self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)

        self.content_conv = nn.Sequential(
            Inception(opt.embedding_dim, incept_dim),
            Inception(incept_dim, incept_dim),
            nn.MaxPool1d(opt.max_seq_len)
        )
        linear_hidden_size = getattr(opt, "linear_hidden_size", 2000)
        self.fc = nn.Sequential(
            nn.Linear(incept_dim, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
            # nn.Linear(linear_hidden_size, opt.label_size)
        )
        self.properties.update(
            {"linear_hidden_size": linear_hidden_size,
             "incept_dim": incept_dim,
             })

    def forward(self, content):

        content_out = self.content_conv(content.permute(0, 2, 1))
        out = content_out.view(content_out.size(0), -1)
        out = self.fc(out)
        return out

