import peft
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

        from yolov5_motion.config import my_config

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
        self.start_conv = ResConv(my_config.model.num_input_channels, 3)

    def forward(self, intial_img, x):
        x = self.start_conv(x)
        x = x + intial_img
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

    def train(self, mode: bool = True):
        for param in self.parameters():
            param.requires_grad = True
        return super().train(mode)


class ControlNetModelLora(nn.Module):
    def __init__(self, yolo_model: YOLOv5Model):
        super().__init__()
        # Clone YOLOv5 backbone structure
        from yolov5_motion.config import my_config

        nodes = nn.ModuleList([module for module in yolo_model.model[:10]])

        initial_num_ch = nodes[0].conv.out_channels
        initial_nub_blocks = len(yolo_model.model[:10][2].m)

        self.nodes = nn.ModuleList(
            [
                Conv(c1=3, c2=initial_num_ch, k=6, s=2, p=2),
                Conv(c1=initial_num_ch, c2=initial_num_ch * 2, k=3, s=2, p=1),
                C3(c1=initial_num_ch * 2, c2=initial_num_ch * 2, n=initial_nub_blocks),
                Conv(c1=initial_num_ch * 2, c2=initial_num_ch * 4, k=3, s=2, p=1),
                C3(c1=initial_num_ch * 4, c2=initial_num_ch * 4, n=initial_nub_blocks * 2),
                Conv(c1=initial_num_ch * 4, c2=initial_num_ch * 8, k=3, s=2, p=1),
                C3(c1=initial_num_ch * 8, c2=initial_num_ch * 8, n=initial_nub_blocks * 3),
                Conv(c1=initial_num_ch * 8, c2=initial_num_ch * 16, k=3, s=2, p=1),
                C3(c1=initial_num_ch * 16, c2=initial_num_ch * 16, n=initial_nub_blocks),
                SPPF(c1=initial_num_ch * 16, c2=initial_num_ch * 16),
            ]
        )

        for my_node, yolo_node in zip(self.nodes, nodes):
            share_weights_recursive(yolo_node, my_node)

        config = peft.LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=["conv"],
            bias="none",
        )
        self.nodes = peft.get_peft_model(self.nodes, config)
        # Convs for ControlNet
        self.convs = nn.ModuleList(
            [
                nn.Sequential(ResConv(initial_num_ch * 4, initial_num_ch * 4)),
                nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), ResConv(initial_num_ch * 4, initial_num_ch * 4)),
                nn.Sequential(ResConv(initial_num_ch * 8, initial_num_ch * 8)),
                nn.Sequential(ResConv(initial_num_ch * 8, initial_num_ch * 8)),
                nn.Sequential(
                    nn.Tanh(),
                ),
                nn.Sequential(
                    nn.Tanh(),
                ),
            ]
        )

        self.start_conv = ResConv(my_config.model.num_input_channels, 3)

    def forward(self, intial_img, x):
        if isinstance(self.nodes, peft.PeftModel):
            model = self.nodes.base_model.model
        else:
            model = self.nodes

        x = self.start_conv(x)
        x = x + intial_img
        # Convs
        x = model[0](x)
        x = model[1](x)
        # C3 + Res Con
        x = model[2](x)
        # Conv
        x = model[3](x)

        ####### 17 out #####
        conv_17_out = x + self.convs[0](x)
        ####################

        # C3
        x = model[4](x)
        #####################

        ####### 18 out #####
        conv_18_out = self.convs[1](x)
        ####################

        # Conv
        x = model[5](x)

        ####### 19 out #####
        conv_19_out = x + self.convs[2](x)
        ####################

        x = model[6](x)

        ####### 20 out #####
        conv_20_out = x + self.convs[3](x)
        ####################

        x = model[7](x)

        x = model[8](x)
        ####### 22 out #####
        conv_22_out = self.convs[4](x)
        ####################

        x = model[9](x)

        ####### 23 out #####
        conv_23_out = self.convs[5](x)
        ####################
        return (conv_17_out, conv_18_out, conv_19_out, conv_20_out, conv_22_out, conv_23_out)

    def train(self, mode: bool = True):
        for name, param in self.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        for param in self.convs.parameters():
            param.requires_grad = True

        for param in self.start_conv.parameters():
            param.requires_grad = True

        self.nodes.eval()
        self.convs.train(mode)
        self.start_conv.train(mode)


def test_control_net():
    from yolov5_motion.config import my_config

    print("Testing ControlNet with YOLOv5...")

    # Load YOLOv5 model
    yolo_model = YOLOv5Model(my_config.model.yolo_cfg)
    print("YOLOv5 model loaded successfully")

    # Test input
    test_input = torch.randn(1, 3, 640, 640)

    # Create ControlNetModel
    control_net = ControlNetModelLora(yolo_model)
    control_net.train()
    for name, param in control_net.named_parameters():
        if param.requires_grad:
            print(name)
    print("ControlNetModel created successfully")

    # Test forward pass
    # with torch.no_grad():
    outputs = control_net(test_input, test_input)
    loss = sum(output.sum() for output in outputs)
    loss.backward()

    print(f"ControlNet output shape: {[output.shape for output in outputs]}")
    print("ControlNet test complete.")


def share_weights_recursive(source_module, target_module, prefix=""):
    """
    Recursively share weights between corresponding modules of any depth.

    Args:
        source_module: The module to copy weights from
        target_module: The module to share weights with
        prefix: Parameter name prefix for nested modules (used internally)
    """
    # Ensure modules are compatible
    if type(source_module) != type(target_module):
        print(f"Warning: Module types don't match at {prefix}: {type(source_module)} vs {type(target_module)}")
        return

    # Share parameters at current level
    for name, param in source_module._parameters.items():
        if param is not None:
            if name in target_module._parameters:
                target_module._parameters[name] = param

    # Share buffers (for BatchNorm running stats, etc.)
    for name, buf in source_module._buffers.items():
        if buf is not None:
            if name in target_module._buffers:
                target_module._buffers[name] = buf

    # Recursively share for all child modules
    for name, module in source_module._modules.items():
        if name in target_module._modules:
            new_prefix = f"{prefix}.{name}" if prefix else name
            share_weights_recursive(module, target_module._modules[name], new_prefix)


if __name__ == "__main__":
    test_control_net()

    print("All tests completed!")
