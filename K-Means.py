from sklearn.cluster import KMeans
import numpy as np
import math
import matplotlib.pyplot as plt

def generate_data():
    xs = [-2.5]
    ys = [0]
    xs.append(0)
    ys.append(0)
    circle_point_num = 720
    R = 0.5
    for i in range(circle_point_num):
        theta = 360.0 / circle_point_num * i
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        xs.append(x)
        ys.append(y)
    xs.append(2.0)
    ys.append(0.0)
    for i in range(circle_point_num):
        theta = 360.0 / circle_point_num * i
        x = R * np.cos(theta) + 2.0
        y = R * np.sin(theta)
        xs.append(x)
        ys.append(y)

    return xs, ys


def show_scatter(data, label):
    N = len(data)
    r0 = 0.6
    print(np.shape(label))
    xs_1 = data[0][label == 1]
    print('the shape of xs_1 is ', len(xs_1))
    ys_1 = data[1][label == 1]
    xs_2 = data[0][label == 2]
    print('the shape of xs_2 is ', len(xs_2))
    ys_2 = data[1][label == 2]
    area = (20 * np.random.rand(N)) ** 2  # 0 to 10 point radii
    c = np.sqrt(area)
    # plt.scatter(x, y, marker='^')
    plt.scatter(xs_1, ys_1, c='green', linewidths=0.001)
    plt.scatter(xs_2, ys_2, c='red', linewidths=0.001)
    # Show the boundary between the regions:
    plt.axis([-3, 3, -3, 3])
    plt.show()

if __name__ == '__main__':
    print(np.sin(math.pi))
    xs, ys = generate_data()
    data = np.concatenate([np.expand_dims(xs, 0), np.expand_dims(ys, 0)], axis=0)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, init=np.asarray([[0.0, -2.51], [0, 0]])).fit(np.transpose(data, [1, 0]))
    # kmeans = KMeans(n_clusters=2, init=np.asarray([[0.0, 0.0], [2.0, 0.0]])).fit(np.transpose(data, [1, 0]))
    print(kmeans.labels_)
    label = kmeans.labels_ + 1
    # label = np.squeeze(np.random.randint(1, 3, [len(data[0]), 1]))
    show_scatter(data, label)
