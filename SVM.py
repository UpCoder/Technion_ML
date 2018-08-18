import numpy as np


class function_z:
    def __init__(self, X=None, Y=None):
        self.X = X
        self.Y = Y
        if self.X is not None and self.Y is not None:
            self.Z = self.func(self.X, self.Y)

    def compute(self, x, y):
        self.X = x
        self.Y = y
        self.Z = self.func(self.X, self.Y)
        return self.Z

    def func(self, X, Y):
        return -20 * (X / 2. - X * X - Y * Y) * np.exp(-X * X - Y * Y)
        # return X*Y
    def compute_gradient(self):
        gradient_X = -20*(0.5 - 2 * self.X) * np.exp(-self.X * self.X - self.Y * self.Y) \
                     + 40 * self.X * (self.X / 2. - self.X * self.X - self.Y * self.Y) \
                       * np.exp(-self.X * self.X - self.Y * self.Y)
        gradient_Y = 40 * self.Y * np.exp(-self.X * self.X - self.Y * self.Y) \
                     + 40 * self.Y * (self.X / 2. - self.X * self.X - self.Y * self.Y) \
                       * np.exp(-self.X * self.X - self.Y * self.Y)
        return [gradient_X, gradient_Y]
        # return gradient_X + gradient_Y

def gradient_descent(function_object, init_value, lr):
    cur_x = init_value[0]
    cur_y = init_value[1]
    xs = []
    ys = []
    zs = []
    iter_count = 0
    while True:
        cur_z = function_object.compute(cur_x, cur_y)
        # cur_gradient = function_object.compute_gradient()
        # cur_z = cur_z - lr * cur_gradient
        # if np.abs(lr * cur_gradient) < 1e-5:
        #     break
        gradient_x, gradient_y = function_object.compute_gradient()
        cur_x = cur_x - lr * gradient_x
        cur_y = cur_y - lr * gradient_y
        xs.append(cur_x)
        ys.append(cur_y)
        zs.append(cur_z)
        iter_count += 1
        if np.abs(lr * gradient_x) < 1e-7 and np.abs(lr * gradient_y) < 1e-7:
            break
        if iter_count > 1e+7:
            break
    print('Iter is ', iter_count, ' the minimum value is ', cur_z, ' The point locate at ', cur_x, ' , ', cur_y)
    plot3D_line(np.asarray(xs), np.asarray(ys) ,np.asarray(zs))

def plot3D_line(X, Y, Z):
    # 载入模块
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    # 生成数据

    # 创建 3D 图形对象
    fig = plt.figure()
    ax = Axes3D(fig)

    # 绘制线型图
    ax.plot(X, Y, Z)
    ax.scatter(X, Y, Z)
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')  # 给三个坐标轴注明
    # 显示图
    plt.show()

def plot3D_surface(X, Y, Z):
    import matplotlib.pyplot as plt  # 绘图用的模块
    from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
    fig1 = plt.figure()  # 创建一个绘图对象
    ax = Axes3D(fig1)  # 用这个绘图对象创建一个Axes对象(有3D坐标)
    plt.title("Question 5")  # 总标题
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)  # 用取样点(x,y,z)去构建曲面
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')  # 给三个坐标轴注明
    plt.show()  # 显示模块中的所有绘图对象

def generate_data_mesh():
    x = np.arange(-3, 3, 0.1)
    print(x)
    y = np.arange(-3, 3, 0.1)
    print(y)
    X, Y = np.meshgrid(x, y)
    print(X)
    print(Y)
    print(X.shape)
    function_object = function_z(X, Y)
    Z = function_object.Z
    print('the minimum of Z is ', np.min(Z))
    print('the maximum of Z is ', np.max(Z))
    plot3D_surface(X, Y, Z)


if __name__ == '__main__':
    # generate_data_mesh()
    # init_x = 0.1
    # init_y = 1
    # lr = 0.01
    init_x = 1.5
    init_y = -1
    lr = 0.01
    function_object = function_z(init_x, init_y)
    gradient_descent(function_object, [init_x, init_y], lr)

