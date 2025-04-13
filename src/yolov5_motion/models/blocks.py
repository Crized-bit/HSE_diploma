# Zero Convolution module for ControlNet. Approved

import torch.nn as nn
import torch
import copy

# Import Ultralytics YOLOv5
import sys

sys.path.append("/home/jovyan/p.kudrevatyh/yolov5")

from models.yolo import Model as YOLOv5Model
from models.common import Conv
from utils.torch_utils import initialize_weights
from utils.general import check_img_size


class ZeroConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0)
        # nn.init.zeros_(self.conv.weight)
        # nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return x + self.conv(x)


class ControlNetModel(nn.Module):
    def __init__(self, yolo_model: YOLOv5Model):
        super().__init__()
        # Clone YOLOv5 backbone structure
        self.backbone = nn.ModuleList([copy.deepcopy(module) for module in yolo_model.model[:10]])
        self.convs_idx = [4, 6, 9]

        # Convs for ControlNet
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    ZeroConv(192, 192),
                    nn.SiLU(),
                    ZeroConv(192, 192),
                    nn.SiLU(),
                    ZeroConv(192, 192),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    ZeroConv(384, 384),
                    nn.SiLU(),
                    ZeroConv(384, 384),
                    nn.SiLU(),
                    ZeroConv(384, 384),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    ZeroConv(768, 768),
                    nn.SiLU(),
                    ZeroConv(768, 768),
                    nn.SiLU(),
                    ZeroConv(768, 768),
                    nn.SiLU(),
                ),
            ]
        )

    def forward(self, x):
        # Get features from main model backbone
        conv_outputs = []
        i = 0
        for idx, m in enumerate(self.backbone):
            x = m(x)  # run

            if idx in self.convs_idx:
                conv_output = self.convs[i](x)
                conv_outputs.append(conv_output)
                i += 1
              # save output
        return conv_outputs


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
