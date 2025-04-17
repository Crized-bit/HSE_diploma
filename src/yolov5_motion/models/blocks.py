# Zero Convolution module for ControlNet. Approved

import torch.nn as nn
import torch

# Import Ultralytics YOLOv5
import sys

sys.path.append("/home/jovyan/p.kudrevatyh/yolov5")

from models.yolo import Model as YOLOv5Model  # type: ignore
from models.common import Conv, C3, SPPF  # type: ignore
from utils.torch_utils import initialize_weights  # type: ignore
from utils.general import check_img_size  # type: ignore


class ResConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        if self.c1 == self.c2:
            return x + self.act(self.bn(self.conv(x)))
        else:
            return self.act(self.bn(self.conv(x)))


class ControlNetModel(nn.Module):
    def __init__(self, yolo_model: YOLOv5Model):
        super().__init__()
        # Clone YOLOv5 backbone structure
        self.backbone = nn.ModuleList(
            [
                Conv(c1=3, c2=48, k=6, s=2, p=2),
                Conv(c1=48, c2=96, k=3, s=2, p=1),
                C3(c1=96, c2=96, n=2),
                Conv(c1=96, c2=192, k=3, s=2, p=1),
                C3(c1=192, c2=192, n=4),
                Conv(c1=192, c2=384, k=3, s=2, p=1),
            ]
        )
        nodes = [module.state_dict() for module in yolo_model.model[:10]]
        for my_node, yolo_node in zip(self.backbone, nodes):
            if not isinstance(my_node, nn.Identity):
                my_node.load_state_dict(yolo_node)

        # Convs for ControlNet
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    ResConv(192, 192),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    ResConv(192, 192),
                ),
                nn.Sequential(nn.Identity()),
                nn.Sequential(nn.Identity()),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    ResConv(384, 768),
                ),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    ResConv(384, 768),
                ),
            ]
        )

    def forward(self, x):
        # Convs
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        # C3 + Res Con
        x = x + self.backbone[2](x)
        # Conv
        x = self.backbone[3](x)

        ####### 17 out #####
        conv_17_out = x + self.convs[0](x)
        ####################

        # C3
        x = x + self.backbone[4](x)
        #####################

        ####### 18 out #####
        conv_18_out = self.convs[1](x)
        ####################

        # Conv
        x = self.backbone[5](x)

        ####### 19 out #####
        conv_19_out = x + self.convs[2](x)
        ####################

        ####### 20 out #####
        conv_20_out = x + self.convs[3](x)
        ####################

        ####### 22 out #####
        conv_22_out = self.convs[4](x)
        ####################

        ####### 23 out #####
        conv_23_out = self.convs[5](x)
        ####################
        return (conv_17_out, conv_18_out, conv_19_out, conv_20_out, conv_22_out, conv_23_out)


def test_control_net():
    print("Testing ControlNet with YOLOv5...")

    # Load YOLOv5 model
    try:
        yolo_model = YOLOv5Model("/home/jovyan/p.kudrevatyh/yolov5/models/yolov5m.yaml")
        print("YOLOv5 model loaded successfully")

        # Test input
        test_input = torch.randn(1, 3, 640, 640)

        # Create ControlNetModel
        control_net = ControlNetModel(yolo_model)
        print("ControlNetModel created successfully")

        # Test forward pass
        # with torch.no_grad():
        outputs = control_net(test_input)
        loss = sum(output.sum() for output in outputs)
        loss.backward()

        print(f"ControlNet output shape: {[output.shape for output in outputs]}")
        print("ControlNet test complete.")

    except Exception as e:
        print(f"Error in ControlNet test: {e}")


if __name__ == "__main__":
    test_control_net()

    print("All tests completed!")
