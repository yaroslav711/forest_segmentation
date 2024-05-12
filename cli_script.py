import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model


@tf.keras.utils.register_keras_serializable()
def iou(y_true, y_pred, smooth=1e-6):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)

    intersection = K.sum(y_true_flat * y_pred_flat)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


segm_model = load_model('./models/segmentation_model.keras', custom_objects={'iou': iou})
class_model = load_model('./models/classification_model.keras')

classes = ['coniferous', 'deciduous', 'mixed']
classes_ru = {
    'coniferous' : 'Хвойный',
    'deciduous': 'Лиственный',
    'mixed' : 'Смешанный'
}

image_path = input('Введите путь до исходного спутникового снимка квадратного формата с расширением .png: ')
result_path = image_path.replace('.png', '_result.png')
image = plt.imread(image_path).reshape((1, 256, 256, 3))
scale = int(input('Введите масштаб снимка (одно целое значение для обеих осей в метрах): '))

segm = segm_model.predict(image).reshape((256, 256, 1))
class_ = classes[np.argmax(
    class_model.predict(image)
)]


avg_distances = {
    'coniferous': [3.0, 3.9],
    'deciduous': [5.1, 6.8],
    'mixed': [4.1, 5.7]
}

d2n = pd.read_csv('./dataset/distance2number.csv', delimiter=';')


# Плотность леса
density = segm.mean()

# средние расстояния между деревьями (интервал в м)
avg_distance = avg_distances[class_]

# общая площадь на изображении (га)
area = (scale ** 2) / 10000

# площадь леса на изображении (га)
forest_area = area * density

# кол-во деревьев на га (интервал)
hect_trees_num = [
    d2n[d2n['distance'] <= max(avg_distance)]['number'].min(),
    d2n[d2n['distance'] >= min(avg_distance)]['number'].max()
]

# общее кол-во деревьев на изображениие (интервал)
trees_num = np.round(np.dot(hect_trees_num, forest_area))

def ax_decorate_box(ax):
    [j.set_linewidth(0) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom=False, top=False, \
               labelbottom=False, left=False, right=False, labelleft=False)
    return ax

fig, AX = plt.subplots(1, 2, figsize=(8, (8-0.2)/2))
plt.subplots_adjust(0, 0.3, 1, 0.85, hspace=0, wspace=0.01)
for ax in AX:
    ax = ax_decorate_box(ax)

AX[0].imshow(image.reshape((256, 256, 3)), cmap=plt.cm.jet)
AX[1].imshow(segm, cmap=plt.cm.jet)

AX[0].set_title("Оригинал", fontsize=14)
AX[1].set_title("Результат сегментации", fontsize=14)

fig.text(0.5, 0.0, f"""
Плотность посадки леса: {round(density*100)}%;
Тип леса: {classes_ru[class_]};
Среднее расстояние между дереьвями: от {min(avg_distance)}м до {max(avg_distance)}м;
Общая площадь на изображении: {round(area, 1)}га;
Площадь леса на изображении: {round(forest_area, 1)}га;
Количество деревьев на изображении: от {round(min(trees_num))} до {round(max(trees_num))}
""", ha='center')

fig.savefig(
    result_path
)

print(f"Результат анализа успешно сохранен по пути: {result_path}")