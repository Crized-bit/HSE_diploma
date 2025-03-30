import torch
import torch.nn as nn
from copy import deepcopy
import sys

sys.path.append("/home/jovyan/p.kudrevatyh/yolov5")

from models.common import Conv  # Импорт сверточных блоков YOLOv5
from utils.plots import feature_visualization
from ultralytics.nn.tasks import DetectionModel  # Импорт YOLO модели от Ultralytics
from ultralytics import YOLO


# Кастомный слой zero convolution
class ZeroConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(ZeroConv, self).__init__()
        # Обычный слой с 1x1 сверткой
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0)

        # Инициализируем веса и смещения как нули
        self.conv.weight.data.fill_(0)  # Инициализация весов нулями
        self.conv.bias.data.fill_(0)  # Инициализация смещений нулями

    def forward(self, x):
        return self.conv(x)


# Модель ControlNet с zero_convolution
class ControlNet(nn.Module):
    def __init__(self, base_model_encoder: list[nn.Module], in_channels=3, base_channels=64):
        super().__init__()

        # Извлекаем блоки энкодера из YOLOv5
        self.encoder = nn.Sequential(*base_model_encoder)  # Извлекаем энкодер

        # Дополнительные слои для извлечения признаков
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(768, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Zero convolution для карты движения
        self.zero_conv = ZeroConv(in_channels=3, out_channels=3, kernel_size=1)  # Используем кастомный слой

    def forward(self, img, motion_map):
        # Применяем zero convolution к motion_map
        motion_map_transformed = self.zero_conv(motion_map)  # Преобразуем motion_map

        # Суммируем img и преобразованный motion_map
        x = img + motion_map_transformed

        # Проходим через энкодер YOLOv5
        x = self.encoder(x)

        # Дополнительное извлечение признаков
        x = self.feature_extractor(x)

        return x


# Модель YOLOv5 с ControlNet
class YOLOv5WithControlNet(DetectionModel):
    def __init__(self, cfg="yolov5m.yaml", ch=3, nc=None):
        super().__init__(cfg, ch, nc)

        self.controlnet = ControlNet([deepcopy(child) for child in list(self.model.children())[:11]])  # Загружаем ControlNet

    def forward(self, x, control_image=None, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, control_image, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, control_image, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    # def forward(self, img, motion_map):
    #     # Получаем признаки из ControlNet
    #     controlnet_features = self.controlnet(img, motion_map)

    #     # Пропускаем через YOLOv5
    #     x = img
    #     for i, layer in enumerate(list(self.yolov5_model.model.children())[0].children()):
    #         x = layer(x)
    #         # if i == 4:  # Вставляем ControlNet-фичи после определенного слоя (например, после C3)
    #         #     x = x + controlnet_features  # Сложение признаков ControlNet с YOLO

    #     return x


if __name__ == "__main__":
    # Тестирование
    model = YOLOv5WithControlNet(cfg="yolov5m.yaml", ch=3, nc=80)

    # Создаем входные данные
    img = torch.randn(1, 3, 640, 640) * 510 - 255  # Пример входного изображения
    motion_map = torch.randn(1, 3, 640, 640) * 510 - 255  # Пример карты движения

    # Прогоняем через модель
    output = model(img, motion_map)

    print(output.shape)  # Выводим размерность выходных данных
