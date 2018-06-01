# -*- coding: utf-8 -*-
import unittest
import torch
from torch.autograd import Variable
import time
from torchsummary import summary

from objdet_darknet import utils, darknet


class TestUtils(unittest.TestCase):
    def test_config_parser(self):
        blocks = utils.config_parser('../cfg/yolov3.cfg')

    def test_create_modules(self):
        blocks = utils.config_parser('../cfg/yolov3.cfg')
        utils.create_modules(blocks)


class TestDarkNet(unittest.TestCase):
    def test_forward(self):
        net = darknet.Darknet('../cfg/yolov3.cfg')
        channels, H, W = 3, 416, 416
        input_var = Variable(torch.randn(1, channels, H, W))
        # summary(net, input_size=(channels, H, W))
        start_time = time.time()
        output_var = net(input_var)
        print('output_var.shape:', output_var.shape)
        end_time = time.time()
        print('forward time:', end_time - start_time)
