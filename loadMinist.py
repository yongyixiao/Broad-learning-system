# encoding:utf-8

import numpy as np
import struct
import matplotlib.pyplot as plt

train_images_idx3_ubyte_file = 'D:/BroadLearing/MINIST/train-images.idx3-ubyte'

train_labels_idx1_ubyte_file = 'D:/BroadLearing/MINIST/train-labels.idx1-ubyte'

test_images_idx3_ubyte_file = 'D:/BroadLearing/MINIST/t10k-images.idx3-ubyte'

test_labels_idx1_ubyte_file = 'D:/BroadLearing/MINIST/t10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file,'rb').read()

    #解析文件头信息
    offset = 0
    fmt_header = '>IIII'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header,bin_data,offset)
    print('魔数:%d, 图片数量:%d张, 图片大小:%d*%d'% (magic_number,num_images,num_rows,num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print("offset:",offset)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images,num_rows*num_cols)) #[60000,784]
    for i in range(num_images):
        if (i+1) % 10000 == 0:
            print("已解析 %d张" % (i+1))
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape(num_cols*num_rows)
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file,'rb').read()
    offset = 0
    fmt_header = '>II'
    magic_number, num_images = struct.unpack_from(fmt_header,bin_data,offset)
    print('魔数:%d, 图片数量:%d'%(magic_number,num_images))

    # 加载数据
    offset += struct.calcsize(fmt_header)
    labels = np.zeros((num_images,10))
    fmt_image = '>B'
    for i in range(num_images):
        if (i+1)%10000 == 0:
            print('已解析%d 张'% (i+1))
        index = struct.unpack_from(fmt_image,bin_data,offset)[0]
        labels[i][index] = 1
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file = train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_train_labels(idx_ubyte_file = train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(idx_ubyte_file = test_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(idx_ubyte_file = test_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)

"""
def run():
    train_images = load_train_images()
    train_labels = load_train_labels()

    fig,ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()

    for i in range(10):
        print(train_labels[i])
        ax[i].imshow(train_images[i].reshape(28,28),cmap='gray', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run()
"""
ima = load_train_labels()[4]
print(ima)