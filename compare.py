import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Фиксированный базовый путь
BASE_PATH = "/home/jovyan/p.kudrevatyh/yolov5_motion/a100_training_outputs"

# Здесь вы можете указать названия моделей для сравнения
MODEL_NAMES = [
    "base_model",  # Замените на свои модели
    # "bg_sub_new",
    # "bg_sub_lora",
    "lora",
    # "bg_sub_0.2",
    "bg_sub_0.2_lora",
    # Добавьте больше моделей при необходимости
]


def load_metrics(model_name):
    """Загружает метрики из JSON-файла для указанной модели."""
    json_path = os.path.join(BASE_PATH, model_name, "test_metrics", "test_results.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        # Возвращаем объект metrics из JSON
        if "metrics" in data:
            return data["metrics"]
        else:
            print(f"Ошибка: Объект 'metrics' не найден в {json_path}")
            return None
    except FileNotFoundError:
        print(f"Ошибка: Файл {json_path} не найден")
        return None
    except json.JSONDecodeError:
        print(f"Ошибка: Невозможно декодировать JSON из {json_path}")
        return None


def plot_metrics():
    """Создает сравнительную гистограмму метрик для указанных моделей с процентным изменением."""
    # Метрики, которые мы хотим отобразить
    metrics = ["f1", "mAP@0.5", "mAP@0.5:0.95", "precision", "recall"]

    # Количество моделей и метрик
    n_models = len(MODEL_NAMES)
    n_metrics = len(metrics)

    if n_models < 1:
        print("Необходимо указать хотя бы одну модель")
        return

    # Подготовка данных для каждой модели
    all_data = {}
    for model in MODEL_NAMES:
        data = load_metrics(model)
        if data:
            all_data[model] = data

    if not all_data:
        print("Нет данных для отображения")
        return

    # Определение базовой модели (первая в списке)
    base_model = MODEL_NAMES[0]
    if base_model not in all_data:
        print(f"Ошибка: Базовая модель {base_model} не найдена или данные недоступны")
        return

    # Настройка фигуры и сетки
    plt.figure(figsize=(14, 8))
    plt.rcParams.update({"font.size": 12})

    # Ширина столбца
    bar_width = 0.8 / n_models

    # Создание позиций для групп баров
    index = np.arange(n_metrics)

    # Цветовая палитра для моделей
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_models)) if n_models > 1 else ["#8a7fff"]

    # Создание баров для каждой модели
    for i, (model, data) in enumerate(all_data.items()):
        # Получение значений метрик для данной модели
        values = []
        for metric in metrics:
            if metric in data:
                values.append(data[metric])
            elif metric.lower() in data:
                values.append(data[metric.lower()])
            elif metric.replace("@", "_") in data:
                values.append(data[metric.replace("@", "_")])
            else:
                print(f"Метрика {metric} не найдена для модели {model}")
                values.append(0)

        # Создание баров с отступом для каждой модели
        position = index + (i - n_models / 2 + 0.5) * bar_width
        bars = plt.bar(position, values, bar_width, label=model, color=colors[i], alpha=0.7)

        # Добавление значений над барами
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()

            # Вычисление процентного изменения относительно базовой модели
            if model == base_model:
                # Для базовой модели просто отображаем значение
                percentage_text = ""
            else:
                # Получаем значение соответствующей метрики для базовой модели
                base_val = 0
                if metrics[j] in all_data[base_model]:
                    base_val = all_data[base_model][metrics[j]]
                elif metrics[j].lower() in all_data[base_model]:
                    base_val = all_data[base_model][metrics[j].lower()]
                elif metrics[j].replace("@", "_") in all_data[base_model]:
                    base_val = all_data[base_model][metrics[j].replace("@", "_")]

                if base_val > 0:
                    percentage_change = ((val - base_val) / base_val) * 100
                    # Определение цвета изменения (зеленый для положительного, красный для отрицательного)
                    change_color = "green" if percentage_change >= 0 else "red"
                    percentage_text = f"\n({percentage_change:+.1f}%)"
                else:
                    percentage_text = ""

            # Основное значение
            val_text = f"{val:.4f}"

            # Объединяем значение и процентное изменение
            full_text = val_text + percentage_text

            # Цвет текста для процентного изменения
            if model != base_model and percentage_text:
                # Создаем текст со значением
                plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, val_text, ha="center", va="bottom", rotation=0, fontsize=10)

                # Добавляем текст с процентным изменением немного выше
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    percentage_text.strip(),
                    ha="center",
                    va="bottom",
                    rotation=0,
                    fontsize=9,
                    color="green" if "+" in percentage_text else "red",
                )
            else:
                # Для базовой модели просто показываем значение
                plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, val_text, ha="center", va="bottom", rotation=0, fontsize=10)

    # Создание баров для каждой модели
    for i, (model, data) in enumerate(all_data.items()):
        # Получение значений метрик для данной модели
        # Некоторые метрики могут иметь разные имена в JSON
        values = []
        for metric in metrics:
            # Проверяем различные варианты имен метрик в JSON
            if metric in data:
                values.append(data[metric])
            elif metric.lower() in data:
                values.append(data[metric.lower()])
            elif metric.replace("@", "_") in data:
                values.append(data[metric.replace("@", "_")])
            else:
                # Если метрика не найдена, используем 0
                print(f"Метрика {metric} не найдена для модели {model}")
                values.append(0)

        # Создание баров с отступом для каждой модели
        position = index + (i - n_models / 2 + 0.5) * bar_width
        bars = plt.bar(position, values, bar_width, label=model, color=colors[i], alpha=0.7)

        # Добавление значений над барами
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{val:.4f}", ha="center", va="bottom", rotation=0, fontsize=10)

    # Настройка графика
    plt.ylabel("Value", fontweight="bold")
    plt.title("Test Detection Metrics", fontsize=16, fontweight="bold")
    plt.xticks(index, metrics)
    plt.ylim(0, 1.0)  # Устанавливаем предел по оси Y от 0 до 1

    # Добавление сетки
    plt.grid(axis="y", linestyle="-", alpha=0.2)
    plt.grid(axis="x", linestyle="-", alpha=0.2)

    # Добавление легенды с выделением базовой модели
    if n_models > 1:
        # Создаем элементы легенды вручную (по одному для каждой модели)
        custom_handles = []
        custom_labels = []

        for i, model_name in enumerate(MODEL_NAMES):
            if model_name in all_data:  # Только модели с данными
                patch = plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.7)
                custom_handles.append(patch)
                # Добавляем "(base)" к имени базовой модели
                if i == 0:
                    custom_labels.append(f"{model_name} (base)")
                else:
                    custom_labels.append(model_name)

        # Добавляем кастомную легенду
        plt.legend(custom_handles, custom_labels, loc="upper right", title="Models")

    # Настройка фона и рамок
    ax = plt.gca()
    ax.set_facecolor("#f0f0f0")

    # Сохранение и отображение
    output_path = os.path.join(BASE_PATH, "metrics_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"График сохранен: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_metrics()
