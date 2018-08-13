import cv2
import numpy as np
import imageio
from PIL import Image
def show_img(img):
    img = Image.fromarray(np.asarray(img, np.uint8))
    img.show()

def split_img():
    images = []
    total_image_path = '/Users/Liang/Documents/dataset/FaceClassification/face.gif'
    img = imageio.imread(total_image_path)
    (t_w, t_h) = np.shape(img)
    print(t_w, t_h)
    e_w = 57
    e_h = 47
    start_i = 0
    count = 0
    while start_i < t_w:
        start_j = 0
        while start_j < t_h - 2:
            cur_image = img[start_i: start_i + e_w, start_j: start_j + e_h]
            images.append(cur_image)
            start_j += e_h
            # start_j += 1
            cv2.imwrite('./imgs/' + str(count) + '.jpg', cur_image)
            count += 1
        start_i += e_w
        # start_i += 1
    return np.reshape(np.asarray(images), [count, e_w * e_h])

def PCA():

    images = split_img()
    show_img(np.reshape(images[0], [57, 47]))
    print(np.shape(images))
    u_x = np.mean(images, axis=0)
    print('The shape of u_x is ', np.shape(u_x))
    transformed_images = images - np.expand_dims(u_x, 0)
    print('The transformed_image is ', np.shape(transformed_images),
          ' it\'s mean is ', np.sum(np.mean(transformed_images, axis=1)))
    sigma = np.dot(np.transpose(transformed_images), transformed_images)
    print('The shape of sigma is ', np.shape(sigma))
    lambda_, u = np.linalg.eigh(sigma)
    idx = np.argsort(lambda_)[::-1]
    u = u[:, idx]
    lambda_ = lambda_[idx]
    print('The shape of eigenvector is ', np.shape(u))
    print('The shape of eigenvalue is ', np.shape(lambda_))
    show_img(np.reshape(np.dot(np.dot(images[0], u[:, :1000]), np.transpose(u[:, :1000])), [57, 47]))

if __name__ == '__main__':
    PCA()

