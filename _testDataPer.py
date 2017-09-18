# Load pickled data
import pickle

# TODO: fill this in based on where you saved the training and testing data
training_file = './lab 2 data/train.p'
testing_file = './lab 2 data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# TODO: number of training examples
n_train = len(X_train)
n_train1=X_train.size
n_train2=X_train.shape
# TODO: number of testing examples
n_test = len(X_test)
n_test1=X_test.size
n_test2=X_test.shape
# TODO: what's the shape of an image?
image_shape = X_train[0].shape

# TODO: how many classes are in the dataset
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

def showimg(X_train):
    import matplotlib.pyplot as plt
    import numpy as np
    cols=int(np.sqrt(len(X_train)))
    rows=int(np.sqrt(len(X_train)))
    init=0
    for i in np.arange(cols):
        for j in np.arange(rows):
            if j==0:
                img=X_train[init+i*rows]
                continue
            img=np.column_stack((img,X_train[init+i*rows+j]))
        if i==0:
            img1=img
            continue
        img1=np.vstack((img1,img))

    plt.imshow(img1)


    #plt.plot(y_train)
    plt.show()

import scipy.ndimage
import random
import numpy as np
from tqdm import tqdm

images=np.zeros((43,32,32,3),dtype=np.uint8)
for i in np.arange(43):
    a=np.where(y_train==(i))
    images[i]=X_train[a[0][0]]
showimg(images)

features=np.zeros((3*len(X_train),32,32,3),dtype=np.uint8)
labels=np.zeros((3*len(X_train),1),dtype=np.uint8)

for i in tqdm(np.arange(len(X_train))):

    img = X_train[i]
    y = y_train[i]
    features[3 * i] = img
    labels[3 * i] = y

    rotate = random.uniform(-15, 15)
    img1 = scipy.ndimage.rotate(img, rotate, (1, 0), False)
    features[3 * i + 1] = img1
    labels[3 * i + 1] = y

    shiftx = random.randint(-2, 2)
    shifty = random.randint(-2, 2)
    img1=scipy.ndimage.shift(img,[shiftx,shifty,0])
    features[3*i+2]=img1
    labels[3*i+2]=y

    

#rotate=random.uniform(-15, 15)
#NewImg=sss.rotate(X_train,rotate,(2,1),False)


def normalize_greyscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    # ToDo: Implement Min-Max scaling for greyscale image data
    a = 0.1
    b = 0.9
    greyscale_min = 0
    greyscale_max = 255
    return a + ( ( (image_data - greyscale_min)*(b - a) )/( greyscale_max - greyscale_min ) )

features=np.array(features,dtype=np.float32)
features=normalize_greyscale(features)
X_test=np.array(X_test,dtype=np.float32)
X_test=normalize_greyscale(X_test)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample

# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(labels)
labels = encoder.transform(labels)
y_test = encoder.transform(y_test)

# Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
labels = labels.astype(np.float32)
y_test = y_test.astype(np.float32)

train_features, valid_features, train_labels, valid_labels = train_test_split(
    features,
    labels,
    test_size=0.05,
    random_state=832289)

import os
pickle_file = 'data.pickle'
if not os.path.isfile(pickle_file):
    print('Saving data to pickle file...')
    try:
        with open('data.pickle', 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': train_features,
                    'train_labels': train_labels,
                    'valid_dataset': valid_features,
                    'valid_labels': valid_labels,
                    'test_dataset': X_test,
                    'test_labels': y_test,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Data cached in pickle file.')


