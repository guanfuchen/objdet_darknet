# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x):
        pass


class EmptyLayer(nn.Module):
    def forward(self, x):
        pass

    def __init__(self):
        super(EmptyLayer, self).__init__()


def create_modules(blocks):
    module_list = nn.ModuleList()
    net_info = blocks[0]  # 第一层net存储了网络训练信息
    prev_filters = int(net_info['channels'])  # 输入使RGB，通道为3
    output_filters = []
    for block_id, block in enumerate(blocks[1:]):
        module = nn.Sequential()
        if block['type'] == 'convolutional':
            activation = block['activation']
            try:
                bn_flag = int(block['batch_normalize'])
                bias = False
            except:
                bn_flag = 0
                bias = True
            out_channels = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            pad_flag = int(block['pad'])
            if pad_flag:
                padding = (kernel_size-1)//2
            else:
                padding = 0
            conv = nn.Conv2d(prev_filters, out_channels, kernel_size, stride, padding, bias=bias)
            module.add_module('conv_{}'.format(block_id), conv)
            if bn_flag:
                bn = nn.BatchNorm2d(out_channels)
                module.add_module('bn_{}'.format(block_id), bn)
            if activation=='leaky':
                activation_layer = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{}'.format(block_id), activation_layer)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
            module.add_module('upsample_{}'.format(block_id), upsample)
        elif block['type'] == 'route':
            route_layers = block['layers'].split(',')
            start = int(route_layers[0])
            try:
                end = int(route_layers[1])
            except:
                end = 0
            if end > 0:
                end = end - block_id

            assert start < 0
            assert end <= 0  # end==0只有一个start

            # print('start:{},end:{}'.format(start, end))

            route = EmptyLayer()
            module.add_module('route_{}'.format(block_id), route)

            if end < 0:
                out_channels = output_filters[block_id + start] + output_filters[block_id + end]
            else:
                out_channels = output_filters[block_id + start]

        elif block['type'] == 'shortcut':
            # from_block = int(block['from'])
            # activation = block['linear']
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(block_id), shortcut)
        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [int(mask_item) for mask_item in mask]
            anchors = block['anchors'].split(',')
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[mask_item] for mask_item in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(block_id), detection)

        prev_filters = out_channels
        output_filters.append(out_channels)

        module_list.append(module)

    # print(module_list)
    return net_info, module_list


def config_parser(config_file_name):
    """
    :param config_file_name: 配置文件路径
    :return: blocks：包含的配置dict
    """
    config_file_pointer = open(config_file_name, 'rb')
    config_lines = config_file_pointer.read().split('\n')
    config_lines = [config_line for config_line in config_lines if len(config_line) > 0]  # 去除空格行
    config_lines = [config_line.strip() for config_line in config_lines]  # 去除左右空格
    config_lines = [config_line for config_line in config_lines if config_line[0] != '#']  # 去除注释
    # print(config_lines)
    blocks = []
    block = {}
    for config_line in config_lines:
        if config_line[0] == '[':
            if len(block) != 0:  # 先前的block信息
                # print(block)
                blocks.append(block)
                block = {}
            block['type'] = config_line[1:-1]
        else:
            key, value = config_line.split('=')
            key = key.strip()
            value = value.strip()
            block[key] = value
    blocks.append(block)  # 最后一层

    # print(len(blocks))
    return blocks


def predict_transform(prediction, input_dim, anchors, num_classes):
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    prediction[:, :, :4] *= stride

    return prediction
