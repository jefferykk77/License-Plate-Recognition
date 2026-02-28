import os.path
from model import Model
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import cv2


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image)


def picture_division(path):
    license_image = cv2.imread(path)#读图
    gray_image = cv2.cvtColor(license_image, cv2.COLOR_BGR2GRAY)#灰度，特征提取
    # cv2.THRESH_OTSU自动设置最优阈值
    ret, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #二值化，图片变为黑白图像
    result = np.sum(binary_image / 255.0 ,axis=0)# 归一化 0 -> 列

    nums = {}
    cnt = 0
    i = 0
    while i<len(result):
        if result[i] == 0:
            i += 1
        else:
            start = i
            while result[i]!=0:
                i+=1
            end = i
            nums[cnt] = [start,end-1]
            cnt += 1

    if not os.path.exists('temp'):
        os.mkdir('temp')

    characters = []
    for i in range(8):
        if i == 2:
            continue
        start,end = nums[i]
        padding = (170 - (end - start)) // 2
        temp = binary_image[:,start:end]
        padding_image = np.pad(temp, ((0, 0), (padding, padding)), 'constant', constant_values=0)
        resized_image = cv2.resize(padding_image, (32, 32))
        cv2.imwrite(f'temp/{i}.jpg', resized_image)
        characters.append(f'temp/{i}.jpg')
    return characters


def match_labels(label_dict):   # 返回将标签与汉字的映射关系
    temp = {'yun': '云', 'cuan': '川', 'hei': '黑', 'zhe': '浙', 'ning': '宁',
            'yu': '豫', 'ji': '冀', 'hu': '沪', 'jl': '吉', 'sx': '晋', 'lu': '鲁',
            'qing': '青', 'zang': '藏', 'e1': '鄂', 'meng': '蒙', 'gan1': '甘',
            'qiong': '琼', 'shan': '陕', 'min': '闽', 'su': '苏', 'xin': '新',
            'wan': '皖', 'jing': '京', 'xiang': '湘', 'gui': '贵', 'yu1': '渝',
            'jin': '津', 'gan': '赣', 'yue': '粤', 'gui1': '桂', 'liao': '辽'}
    name_dict = {}
    for key, val in label_dict.items():      # 本次转换的目的是转换字母和汉字
        name_dict[key] = temp.get(val, val)
    return name_dict


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = Model().to(device)#加载模型
model.load_state_dict(torch.load('model.pth'))#加载模型权重
model.eval()

IMAGE_PATH = r"./Cache_5b2ed614cefeb2a7.png"  # Update the path here
picture_division(IMAGE_PATH)#传入def

name_dict = match_labels({0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'Q', 25: 'R', 26: 'S', 27: 'T', 28: 'U', 29: 'V', 30: 'W', 31: 'X', 32: 'Y', 33: 'Z', 34: 'cuan', 35: 'e1', 36: 'gan', 37: 'gan1', 38: 'gui', 39: 'gui1', 40: 'hei', 41: 'hu', 42: 'ji', 43: 'jin', 44: 'jing', 45: 'jl', 46: 'liao', 47: 'lu', 48: 'meng', 49: 'min', 50: 'ning', 51: 'qing', 52: 'qiong', 53: 'shan', 54: 'su', 55: 'sx', 56: 'wan', 57: 'xiang', 58: 'xin', 59: 'yu', 60: 'yu1', 61: 'yue', 62: 'yun', 63: 'zang', 64: 'zhe'})

result = ""
for i in range(8):
    if i == 2:
        continue
    image = load_image(f'temp/{i}.jpg').to(device)
    temp = model(image.unsqueeze(0))
    _, predicted = torch.max(temp, 1)
    result += name_dict[predicted.item()]
print(result)
