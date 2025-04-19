import argparse
import os
import sys
import torch
import numpy as np
import json
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

# Добавляем пути для импорта YOLOv5 и наших модулей
yolov5_dir = "/home/jovyan/p.kudrevatyh/yolov5"
if os.path.exists(yolov5_dir):
    sys.path.append(yolov5_dir)
    print(f"Added YOLOv5 directory to path: {yolov5_dir}")
else:
    print(f"Warning: YOLOv5 directory not found at {yolov5_dir}")

# Импортируем функции YOLOv5 для постобработки детекций
try:
    from utils.general import non_max_suppression, scale_boxes  # type: ignore

    print("Successfully imported YOLOv5 utils")
except ImportError as e:
    print(f"Warning: Could not import YOLOv5 modules: {e}")

# Импортируем наши модули
from yolov5_motion.models.yolov5_controlnet import create_combined_model
from yolov5_motion.data.dataset_splits import create_dataset_splits
from yolov5_motion.utils.metrics import calculate_precision_recall, calculate_map
from yolov5_motion.data.utils import draw_bounding_boxes


def parse_args():
    """Анализ аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Test YOLOv5 with ControlNet for Motion Detection")

    # Единственный обязательный аргумент - путь к YAML конфигурации
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")

    # Опциональные аргументы для переопределения конфигурации
    parser.add_argument("--output_dir", type=str, help="Override output directory from config")
    parser.add_argument("--checkpoint", type=str, help="Override checkpoint path from config")
    parser.add_argument("--batch_size", type=int, help="Override batch size from config")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of predictions (overrides config)")
    parser.add_argument("--no_visualize", action="store_true", help="Disable visualization of predictions (overrides config)")

    args = parser.parse_args()
    return args


def load_config(config_path, args):
    """
    Загружает конфигурацию из YAML файла и применяет переопределения из args

    Args:
        config_path: Путь к YAML файлу
        args: Аргументы командной строки для переопределения

    Returns:
        dict: Объединенная конфигурация
    """
    # Загружаем конфигурацию из YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def visualize_predictions(model, dataloader, device, config, output_dir):
    """
    Визуализация предсказаний модели на тестовых данных

    Args:
        model: Модель YOLOv5 с ControlNet
        dataloader: Даталоадер с тестовыми данными
        device: Устройство (cuda/cpu)
        config: Конфигурация тестирования
        output_dir: Директория для сохранения визуализаций
    """
    # Получаем настройки
    conf_thres = config["testing"]["detection"]["conf_thres"]
    iou_thres = config["testing"]["detection"]["iou_thres"]
    max_det = config["testing"]["detection"]["max_det"]
    num_samples = config["testing"]["vis_samples"]

    # Создаём директорию для визуализаций
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Получаем один батч из тестового набора
    batch = next(iter(dataloader))

    current_frames = batch["current_frames"].to(device)
    control_images = batch["control_images"].to(device)
    annotations = batch["annotations"]

    # Ограничиваем количество образцов
    num_samples = min(num_samples, current_frames.size(0))

    print(f"Visualizing {num_samples} test samples...")

    with torch.no_grad():
        # Получаем предсказания
        predictions = model(current_frames[:num_samples], control_images[:num_samples])

        # Применяем NMS для получения финальных детекций
        try:
            detections = non_max_suppression(
                predictions if isinstance(predictions, torch.Tensor) else predictions[0],
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
            )

            # Создаём визуализации для каждого образца
            panel_images = []

            for i in range(num_samples):
                # Конвертируем тензоры в numpy массивы
                frame = current_frames[i].cpu().permute(1, 2, 0).numpy() * 255
                frame = frame.astype(np.uint8)

                control = control_images[i].cpu().permute(1, 2, 0).numpy() * 255
                control = control.astype(np.uint8)

                # Рисуем ground truth аннотации зелёным цветом
                gt_frame = frame.copy()
                if i < len(annotations):
                    gt_frame = draw_bounding_boxes(gt_frame, annotations[i], color=(0, 255, 0))

                # Рисуем предсказанные ограничивающие рамки красным цветом
                pred_frame = frame.copy()
                if i < len(detections) and detections[i] is not None:
                    # Масштабируем координаты к размеру изображения
                    det = detections[i].clone()
                    det[:, :4] = scale_boxes(current_frames.shape[2:], det[:, :4], pred_frame.shape).round()

                    # Рисуем каждую детекцию
                    for *xyxy, conf, cls in det:
                        # Конвертируем координаты в целые числа
                        xyxy = [int(x.item()) for x in xyxy]

                        # Рисуем ограничивающую рамку
                        cv2.rectangle(pred_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)

                        # Добавляем метку с уверенностью
                        label = f"{int(cls.item())}: {conf:.2f}"
                        cv2.putText(pred_frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Создаём слайд из трёх изображений: GT, Predictions, Control Image
                comparison = np.hstack((gt_frame, pred_frame, control))

                # Добавляем заголовки
                header = np.ones((30, comparison.shape[1], 3), dtype=np.uint8) * 255
                cv2.putText(header, "Ground Truth | Predictions | Control Image", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                comparison_with_header = np.vstack((header, comparison))

                # Сохраняем визуализацию
                cv2.imwrite(str(vis_dir / f"test_sample_{i}.jpg"), cv2.cvtColor(comparison_with_header, cv2.COLOR_RGB2BGR))

                # Добавляем в панель
                if not panel_images:
                    panel_images.append(comparison_with_header)
                else:
                    panel_images.append(comparison)

            # Создаём итоговую панель со всеми визуализациями
            if panel_images:
                # Находим максимальную ширину
                max_width = max(img.shape[1] for img in panel_images)

                # Создаём вертикальный стек с выравниванием по ширине
                panel = []
                for img in panel_images:
                    if img.shape[1] < max_width:
                        pad_width = max_width - img.shape[1]
                        pad = np.ones((img.shape[0], pad_width, 3), dtype=np.uint8) * 255
                        padded_img = np.hstack((img, pad))
                        panel.append(padded_img)
                    else:
                        panel.append(img)

                if panel:
                    panel = np.vstack(panel)
                    cv2.imwrite(str(vis_dir / "test_panel.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                    print(f"Visualization panel saved to {vis_dir}/test_panel.jpg")

        except Exception as e:
            print(f"Error during visualization: {e}")


def plot_test_metrics(test_metrics, output_dir):
    """
    Визуализация метрик тестирования

    Args:
        test_metrics: Словарь с метриками тестирования
        output_dir: Директория для сохранения графиков
    """
    # Создаём директорию для метрик
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Стиль для графиков
    plt.style.use("ggplot")

    # Создаём словарь с метриками для отображения
    metrics_to_plot = {
        "precision": test_metrics.get("precision", 0),
        "recall": test_metrics.get("recall", 0),
        "f1": test_metrics.get("f1", 0),
        "mAP@0.5": test_metrics.get("mAP@0.5", 0),
        "mAP@0.5:0.95": test_metrics.get("mAP@0.5:0.95", 0),
    }

    # Сортируем метрики
    metrics_to_plot = {k: v for k, v in sorted(metrics_to_plot.items())}

    # Строим столбцовую диаграмму для метрик детекции
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_to_plot.keys(), metrics_to_plot.values(), color="royalblue", alpha=0.7)

    # Добавляем значения над столбцами
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.4f}", ha="center", va="bottom")

    plt.ylim(0, 1.05)
    plt.title("Test Detection Metrics", fontsize=16, fontweight="bold")
    plt.ylabel("Value")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Сохраняем график
    plt.savefig(str(metrics_dir / "test_detection_metrics.png"), dpi=300)
    plt.savefig(str(metrics_dir / "test_detection_metrics.pdf"))
    plt.close()

    print(f"Test metrics plots saved to {metrics_dir}")


def test(config):
    """
    Тестирование модели на тестовом наборе данных

    Args:
        config: Конфигурация тестирования из YAML файла
    """
    # Получаем параметры из конфигурации
    preprocessed_dir = config["data"]["preprocessed_dir"]
    annotations_dir = config["data"]["annotations_dir"]
    splits_file = config["data"]["splits_file"]
    control_stack_length = config["data"]["control_stack_length"]
    val_ratio = config["data"]["val_ratio"]

    yolo_cfg = config["model"]["yolo_cfg"]
    img_size = config["model"]["img_size"]
    num_classes = config["model"]["num_classes"]

    batch_size = config["testing"]["batch_size"]
    workers = config["testing"]["workers"]
    checkpoint_path = config["testing"]["checkpoint"]
    visualize = config["testing"]["visualize"]

    conf_thres = config["testing"]["detection"]["conf_thres"]
    iou_thres = config["testing"]["detection"]["iou_thres"]
    max_det = config["testing"]["detection"]["max_det"]

    device_name = config["testing"]["device"]

    disable_controlnet = config["testing"]["disable_controlnet"]

    epoch_num = config["testing"]["epoch_num"]

    # Создаём директорию для результатов
    output_dir = Path(checkpoint_path) / f"test_metrics{epoch_num if epoch_num != 0 else ''}"
    output_dir.mkdir(parents=False, exist_ok=False)

    # Сохраняем используемую конфигурацию
    with open(output_dir / "test_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Выбираем устройство (GPU/CPU)
    device = torch.device("cuda" if (torch.cuda.is_available() and device_name == "cuda") else "cpu")
    print(f"Using device: {device}")

    # Создаём датасеты и даталоадеры
    print("\nCreating datasets and dataloaders...")
    datasets = create_dataset_splits(
        preprocessed_dir=preprocessed_dir,
        annotations_dir=annotations_dir,
        splits_file=splits_file,
        val_ratio=val_ratio,
        augment=False,  # Без аугментации для тестирования
        control_stack_length=control_stack_length,
    )

    # Создаём даталоадер только для тестового набора
    from torch.utils.data import DataLoader
    from yolov5_motion.data.dataset import collate_fn

    test_dataloader = DataLoader(
        datasets["test"], batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True
    )

    print(f"Test dataloader: {len(test_dataloader)} batches with {batch_size} samples per batch")

    # Загружаем модель
    print("\nLoading model...")

    # Путь к чекпоинту
    if not epoch_num:
        checkpoint_path = Path(checkpoint_path) / "checkpoints/best_model.pt"
    else:
        checkpoint_path = Path(checkpoint_path) / f"checkpoints/checkpoint_epoch_{epoch_num}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Создаём модель
    model = create_combined_model(
        cfg=yolo_cfg,
        yolo_weights=None,  # Веса загрузим из чекпоинта
        controlnet_weights=None,  # Веса загрузим из чекпоинта
        img_size=img_size,
        nc=num_classes,
    )

    # Загружаем веса из чекпоинта
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            print(f"Loaded model from checkpoint (epoch {checkpoint['epoch']})")
    else:
        # Пробуем загрузить напрямую как словарь состояния
        try:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from checkpoint")
        except Exception as e:
            print(f"Error loading weights: {e}")
            # Пробуем загрузить как модель, а не состояние модели
            if "model" in checkpoint:
                try:
                    model.load_state_dict(checkpoint["model"])
                    print(f"Loaded model weights from checkpoint['model']")
                except Exception as e:
                    print(f"Error loading weights from checkpoint['model']: {e}")
                    raise ValueError("Could not load model weights from checkpoint")

    model = model.to(device)

    if disable_controlnet:
        print("Disabling ControlNet...")
        model.use_controlnet = False
        
    # print(model_info(model, verbose=True))
    model.eval()  # Устанавливаем режим оценки

    # Тестирование модели
    print("\n===== Testing Model =====")

    # Списки для хранения предсказаний и ground truth для расчёта метрик
    all_pred_boxes = []
    all_true_boxes = []

    # Инициализируем сборщик метрик
    precisions = []
    recalls = []
    f1s = []

    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc="Testing")

        for batch in pbar:
            # Получаем данные и перемещаем на устройство
            current_frames = batch["current_frames"].to(device)
            control_images = batch["control_images"].to(device)
            targets = batch["annotations"]

            # Прямой проход
            predictions = model(current_frames, control_images)

            # Получаем сырые выходы детекции
            detections = non_max_suppression(
                predictions if isinstance(predictions, torch.Tensor) else predictions[0],
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
            )

            # Обрабатываем каждое изображение в батче
            for i, det in enumerate(detections):
                pred_boxes = []
                if det is not None and len(det):
                    # Масштабируем рамки к исходному размеру изображения
                    det[:, :4] = scale_boxes(current_frames.shape[2:], det[:, :4], (640, 640)).round()

                    # Конвертируем в формат, ожидаемый функциями метрик
                    for *xyxy, conf, cls_id in det:
                        # We need to pred only people
                        if cls_id == 0:
                            pred_boxes.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item(), cls_id.item()])

                # Обрабатываем ground truth
                true_boxes = []
                if i < len(targets):
                    for ann in targets[i]:
                        # Конвертируем из формата center в формат corner
                        bbox = ann["bbox"]
                        x1 = bbox[0] - bbox[2] / 2
                        y1 = bbox[1] - bbox[3] / 2
                        x2 = bbox[0] + bbox[2] / 2
                        y2 = bbox[1] + bbox[3] / 2

                        # Получаем ID класса (предполагаем один класс)
                        cls_id = 0  # По умолчанию первый класс

                        true_boxes.append([x1, y1, x2, y2, cls_id])

                # Рассчитываем precision и recall для этого изображения
                if len(true_boxes) > 0 or len(pred_boxes) > 0:
                    precision, recall, f1 = calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5, conf_threshold=conf_thres)

                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)

                # Сохраняем для расчёта mAP
                all_pred_boxes.append(pred_boxes)
                all_true_boxes.append(true_boxes)

    # Рассчитываем mAP
    map50, map = calculate_map(all_pred_boxes, all_true_boxes)

    # Рассчитываем среднюю precision и recall по всем изображениям
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    # Формируем итоговые метрики
    test_metrics = {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1, "mAP@0.5": map50, "mAP@0.5:0.95": map}

    # Создаём визуализации, если запрошено
    if visualize:
        visualize_predictions(model, test_dataloader, device, config, output_dir)

    # Строим графики метрик
    plot_test_metrics(test_metrics, output_dir)

    # Печатаем результаты тестирования
    print("\nTest Results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # Сохраняем результаты тестирования в файл
    with open(output_dir / "test_results.json", "w") as f:
        json.dump({"metrics": test_metrics, "config": {"model": config["model"], "testing": config["testing"]["detection"]}}, f, indent=4)

    print(f"\nTest results saved to {output_dir}")

    return test_metrics


if __name__ == "__main__":
    # Обрабатываем аргументы командной строки
    args = parse_args()

    # Загружаем конфигурацию
    config = load_config(args.config, args)

    # Запускаем тестирование
    test_metrics = test(config)
