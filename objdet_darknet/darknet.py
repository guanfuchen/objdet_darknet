# -*- coding: utf-8 -*-
import ConfigParser
import torch
from torch import nn
from torch.autograd import Variable
import time

import utils


class Darknet(nn.Module):
    def __init__(self, config_file_name):
        super(Darknet, self).__init__()
        self.blocks = utils.config_parser(config_file_name)
        self.net_info, self.module_list = utils.create_modules(self.blocks)

    def forward(self, x):
        outputs = []
        detections = None
        for block_id, block in enumerate(self.blocks[1:]):
            if block['type'] == 'convolutional':
                x = self.module_list[block_id](x)
            elif block['type'] == 'upsample':
                x = self.module_list[block_id](x)
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

                if end < 0:
                    map1 = outputs[block_id + start]
                    map2 = outputs[block_id + end]
                    x = torch.cat((map1, map2), 1)
                else:
                    x = outputs[block_id + start]
            elif block['type'] == 'shortcut':
                from_block_id = int(block['from'])
                x = outputs[block_id - 1] + outputs[block_id + from_block_id]
            elif block['type'] == 'yolo':
                anchors = self.module_list[block_id][0].anchors
                input_height = int(self.net_info["height"])
                input_width = int(self.net_info["width"])
                assert input_height==input_width
                input_dim = (input_height+input_width)//2
                num_classes = int(block["classes"])
                x = utils.predict_transform(x, input_dim, anchors, num_classes)
                if detections is None:
                    detections = x
                else:
                    detections = torch.cat((detections, x), 1)
            outputs.append(x)
        return detections
