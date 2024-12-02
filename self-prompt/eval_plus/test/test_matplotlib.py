from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import rcParams
# config = {
#     "font.family":'Times New Roman',  # 设置字体类型
#     # "axes.unicode_minus": False #解决负号无法显示的问题
# }
# rcParams.update(config)

def plot_line():
    len_x = [i for i in range(0, 81, 10)]

    # dev_y = [0.476] * 11
    dev_y = [0.640, 0.628, 0.634, 0.64, 0.652, 0.665, 0.646, 0.646, 0.646]
    plt.plot(len_x, dev_y, marker=".")

    py_dev_y = [0.457, 0.494, 0.463, 0.494, 0.506, 0.518, 0.506, 0.5, 0.5]


    plt.plot(len_x, py_dev_y, marker=".")

    plt.xlabel("Extra Prompt Length")
    plt.ylabel("pass@1")
    # plt.title("年龄和薪水的关系")
    x_major_locator = MultipleLocator(10)
    # 把x轴的刻度间隔设置为1，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数

    plt.xlim(-5, 85)

    plt.axvline(50, c='red', linestyle='--')  # 竖线
    # plt.grid(True)  # 添加网格
    plt.legend(['Gemma2-9B', 'ChatGLM3-6B'])

    plt.show()


if __name__ == "__main__":
    plot_line()
