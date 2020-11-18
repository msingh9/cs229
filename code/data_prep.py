# importing libraries
import bz2
import pickle
import numpy as np
import random
import sys
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

augmentation = False
write_train = True
write_test = True
np.random.seed(10000000)
in_resolution = (1536, 1103)
patch_size = (64,64) ; # (row,col)
patch_stride = (16,16)
ann2pixels = (15.11, 21.04)
num_classes = 1
debug = 2
image_is_24bit = re.compile(".*_24bit_.*")

# data directory
data_in_dir = ("../data/raw_images", "../data/anno_info")
data_out_dir = "../data"

# find out if fault is located in the image page
def get_fault_count(image_coord, fault_list):
    # Inputs: image_coord (x1,y1,x2,y2) coordinates of top/left and bottom right corners
    #       : fault_list: List of faults for this image
    x1,y1,x2,y2 = image_coord
    ans = 0
    for x,y in fault_list:
        if x > x1 and y > y1 and x < x2 and y < y2:
            ans += 1
    return ans

def has_fault(image_coord, fault_list):
    if get_fault_count(image_coord, fault_list) > 0:
        return 1
    return 0

# process one image
def process_one_image(imgfile, annfile):
    print (imgfile)
    I = plt.imread(imgfile)
    print (I.shape)
    if image_is_24bit.match(imgfile):
        h, w, _ = I.shape
        I = I[:,:,0]
    else:
        h, w = I.shape

    # remove the bottom
    h -= 80
    I = I[:h,:]
    if debug:
        print (f"Reading file {imgfile} and its annotation {annfile} with shape ({w}, {h})")
    fault_coords = []
    with open(annfile) as csvfile:
        info = csv.reader(csvfile)
        # skip header
        next(info, None)
        for row in info:
            if debug > 3:
                print (row)
            x = int((float(row[-2])*in_resolution[0])/ann2pixels[1])
            y = int((float(row[-1])*in_resolution[1])/ann2pixels[0])
            fault_coords.append((x,y))

    # split the images in patches
    X = []
    Y = []
    C = [] ; #  Count of faults
    A = []
    for i in range((h-patch_size[0])//patch_stride[0] + 1):
        for j in range((w-patch_size[1])//patch_stride[1] + 1):
            x1 = j * patch_stride[1]
            y1 = i * patch_stride[0]
            x2 = x1 + patch_size[1]
            y2 = y1 + patch_size[0]
            X.append(I[y1:y2, x1:x2])
            # check if there is fault in this patch
            Y.append(has_fault((x1,y1,x2,y2), fault_coords))
            C.append(get_fault_count((x1,y1,x2,y2), fault_coords))
            A.append((x1,y1,x2,y2))

    # plot for sample
    if debug > 2:
        fix, ax = plt.subplots(1)
        ax.imshow(I, cmap='gray')
        # pick random samples and plot with thier y value to check
        for i in range(40):
            index = random.randint(0,len(Y))
            x1, y1, x2, y2 = A[index]
            width = x2 - x1
            height = y2 - y1
            c1 = (x1+x2)//2
            c2 = (y1+y2)//2
            rect = patches.Rectangle((x1, y1), height, width, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(c1,c2, str(Y[index]) + '(' + str(C[index]) + ')' )
        plt.scatter([x[0] for x in fault_coords], [x[1] for x in fault_coords], c='r', s=10)
        plt.show()

    return X, Y, C

print (data_in_dir[0])
x_all = []
y_all = []
c_all = []
for (_, __, images) in os.walk(data_in_dir[0]):
    lll = len(images)
    print (f"Found {lll} images")
    for image in images:
        annfile = data_in_dir[1] + "/" + image.split(".")[0] + ".csv"
        imgfile = data_in_dir[0] + "/" + image
        X, Y, C = process_one_image(imgfile, annfile)
        x_all.extend(X)
        y_all.extend(Y)
        c_all.extend(C)

x_all = np.array(x_all)
y_all = np.array(y_all)
c_all = np.array(c_all)
print (x_all.shape, y_all.shape, c_all.shape)
npos = np.sum(y_all == 1)
nneg = np.sum(y_all == 0)
n = len(y_all)
pp = npos*100/n
nn = nneg*100/n
print (f"Positive samples # = {npos} {pp}")
print (f"Negative samples # = {nneg} {nn}")

# count histogram
max_count = np.max(c_all)
print (f"max_count of faults is {max_count}")
#CountHistogram = []
#for i in range(max_count):
#    CountHistogram.append(np.sum(C == i))

# plot histogram of count
if debug > 1:
    fix, ax = plt.subplots(1)
    ax.hist(c_all, bins=max_count)
    plt.show()

## Re-shuffle, split and save
## 80% train+dev and 20% test
indices = np.random.choice(n, size=n, replace=False)
m20p = int(n*0.2)
x_test = x_all[indices[0:m20p]]
y_test = y_all[indices[0:m20p]]
c_test = c_all[indices[0:m20p]]
x_train_all = x_all[indices[m20p:]]
y_train_all = y_all[indices[m20p:]]
c_train_all = c_all[indices[m20p:]]

## further split of 80:20 for train+dev
n = y_train_all.shape[0]
indices = np.random.choice(n, size=n, replace=False)
m20p = int(n*0.2)
x_dev = x_train_all[indices[0:m20p]]
y_dev = y_train_all[indices[0:m20p]]
c_dev = c_train_all[indices[0:m20p]]
x_train = x_train_all[indices[m20p:]]
y_train = y_train_all[indices[m20p:]]
c_train = c_train_all[indices[m20p:]]

# save the data
for ttt in ("test", "train", "dev"):
    fname = data_out_dir + "/" + ttt + "_" + str(patch_size[0]) + "_data.bz2"
    fout = bz2.BZ2File(fname, 'wb')
    print (f"Dumping {ttt} data into {fname}")
    if ttt == "test":
        pickle.dump((x_test, y_test, c_test), fout, protocol=4)
    elif ttt == "train":
        pickle.dump((x_train, y_train, c_train), fout, protocol=4)
    elif ttt == "dev":
        pickle.dump((x_dev, y_dev, c_dev), fout, protocol=4)
    else:
        exit("Something went wrong")

# Randomly select 50 images from test set for manual annotation
if not os.path.isdir(data_out_dir + "/test_images"):
    os.makedirs(data_out_dir + "/test_images")
indices = np.random.choice(50, size=50, replace=False)
i = 0
for index in indices:
    plt.imsave(data_out_dir + "/test_images/image" + str(i) + ".png", x_test[index], cmap='gray')
    i += 1

x_man_test = x_test[indices]
y_man_test = y_test[indices]
c_man_test = c_test[indices]
fname = data_out_dir + "/man_test"  + "_" + str(patch_size[0]) + "_data.bz2"
fout = bz2.BZ2File(fname, 'wb')
print(f"Dumping man_test data into {fname}")
pickle.dump((x_man_test, y_man_test, c_man_test), fout, protocol=4)


