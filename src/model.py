import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.common import Conv  # Импорт сверточных блоков YOLOv5


class ControlNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # Первые сверточные слои (энкодер)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Дополнительные слои для выделения признаков
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Выходной сверточный слой
        self.out_channels = base_channels * 2
        self.output_layer = nn.Conv2d(base_channels * 2, self.out_channels, kernel_size=1)

    def forward(self, img, motion_map):
        x = img + motion_map  # Суммируем входное изображение и карту движения
        x = self.encoder(x)
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        return x


class YOLOv5WithControlNet(nn.Module):
    def __init__(self, yolo_model, controlnet_model):
        super().__init__()
        self.yolo = yolo_model  # Загружаем стандартную YOLOv5
        self.controlnet = controlnet_model  # Загружаем ControlNet

        # Дополнительный 1x1 сверточный слой для приведения числа каналов
        self.controlnet_adapter = nn.Conv2d(controlnet_model.out_channels, 64, kernel_size=1)

    def forward(self, img, motion_map):
        # Обрабатываем вход через ControlNet
        controlnet_features = self.controlnet(img, motion_map)  # (B, C, H, W)

        # Приведение к нужному числу каналов
        controlnet_features = self.controlnet_adapter(controlnet_features)

        # Убеждаемся, что spatial resolution совпадает
        controlnet_features = F.interpolate(controlnet_features, size=img.shape[2:], mode="bilinear", align_corners=False)

        # Получаем промежуточные признаки YOLOv5
        x = img
        for i, layer in enumerate(self.yolo.model):
            x = layer(x)
            if i == 4:  # Вставляем ControlNet-фичи после определенного слоя (например, после C3)
                x = x + controlnet_features  # Сложение фичей ControlNet с YOLO

        return x


# Загрузка моделей (замени на свои пути)
yolo_model = torch.load("yolov5s.pt")  # Загружаем предобученную YOLOv5s
controlnet_model = ControlNet()  # Создаем экземпляр ControlNet

# Создаем расширенную модель
model = YOLOv5WithControlNet(yolo_model, controlnet_model)

# Проверка работы
img = torch.randn(1, 3, 640, 640) * 510 - 255  # Заглушка для входного изображения в диапазоне [-255, 255]
motion_map = torch.randn(1, 3, 640, 640) * 510 - 255  # Заглушка для карты движения
output = model(img, motion_map)

print(output)
