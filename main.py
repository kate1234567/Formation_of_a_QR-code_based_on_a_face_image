import os
import matplotlib.pyplot as plt
import qrcode
from PIL import Image
import cv2 as cv
import numpy as np

def crop_image(image):
    img_width, img_height = image.size
    crop_height, crop_width = 175, 175
    new_image = image.crop(((img_width - crop_height) // 2, (img_height - crop_height) // 2, (img_width + crop_width) // 2, (img_height + crop_height) // 2))
    return new_image

def qr_code_generation(img):
    qr_codes = []
    image = crop_image(img)

    r1, g1, b1 = image.split()
    r2, g2, b2 = image.split()
    r3, g3, b3 = image.split()

    gray_img = image.convert("L")

    qr_info = qrcode.QRCode(version=1, box_size=5, border=1)
    qr_info.add_data("https://cmp.felk.cvut.cz/~spacelib/faces/faces94.html")
    qr_info.make(fit=True)
    qr_info_image = qr_info.make_image(fill_color="black", back_color="white")

    qr_antro = qrcode.QRCode(version=1, box_size=10, border=1)
    points = antro(image)
    qr_antro.add_data(points[0])
    qr_antro.make(fit=True)
    qr_antro_image = qr_antro.make_image(fill_color="black", back_color="white")
    qr_antro_image = qr_antro_image.resize((175, 175))

    r1.paste(gray_img)
    g1.paste(qr_info_image)
    qr_pip = Image.merge("RGB", (r1, g1, b1))
    qr_codes.append(qr_pip)

    r2.paste(gray_img)
    g2.paste(qr_info_image)
    b2.paste(qr_antro_image)
    qr_pia = Image.merge("RGB", (r2, g2, b2))
    qr_codes.append(qr_pia)

    r3.paste(gray_img)
    g3.paste(qr_antro_image)
    b3.paste(qr_info_image)
    qr_pai = Image.merge("RGB", (r3, g3, b3))
    qr_codes.append(qr_pai)

    return qr_codes

def antro(image):
    gray_image = image.convert("L")
    gray_image = np.array(gray_image)
    haarcascade = 'haarcascades/haarcascade_frontalface_alt2.xml'
    detector = cv.CascadeClassifier(haarcascade)
    face = detector.detectMultiScale(gray_image)

    LBFmodel = 'facemark_api/lbfmodel.yaml'
    landmark_detector = cv.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)
    _, landmarks = landmark_detector.fit(gray_image, face)

    return landmarks

def import_img(i, j):
    image = Image.open('faces94/s' + str(i + 1) + '/' + str(j + 1) + '.jpg')
    return image

def show_results():
    results = []
    plt.figure(figsize=(10, 5))

    for i in range(70):
        for j in range(1):
            results = qr_code_generation(import_img(i, j))
            ax1 = plt.subplot(1, 4, 1)
            ax2 = plt.subplot(1, 4, 2)
            ax3 = plt.subplot(1, 4, 3)
            ax4 = plt.subplot(1, 4, 4)

            ax1.clear()
            ax1.imshow(import_img(i, j))
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_xlabel('Класс № ' + str(i + 1))

            ax2.clear()
            ax2.imshow(results[0])
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_xlabel('QRpip')

            ax3.clear()
            ax3.imshow(results[1])
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_xlabel('QRpia')

            ax4.clear()
            ax4.imshow(results[2])
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_xlabel('QRpai')

            plt.pause(2.5)

show_results()

