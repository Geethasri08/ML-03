import numpy as np
import os
from sklearn.svm import SVC
import cv2
from skimage import io
def preprocess_image(file_path):
    img = io.imread(file_path, as_gray=True)
    img = cv2.resize(img, (32, 32))
    img = img.flatten()
    return img
cats_train_folder = "Training _folder/cats"  
cat_images = []
for filename in os.listdir(cats_train_folder):
    img_path = os.path.join(cats_train_folder, filename)
    img = preprocess_image(img_path)
    cat_images.append(img)
dogs_train_folder ="Training _folder/dogs"  
dog_images = []
for filename in os.listdir(dogs_train_folder):
    img_path = os.path.join(dogs_train_folder, filename)
    img = preprocess_image(img_path)
    dog_images.append(img)
X = np.array(cat_images + dog_images)
y = np.array([0] * len(cat_images) + [1] * len(dog_images))
clf = SVC(kernel='linear')
clf.fit(X, y)
sample_image_path ="Test_folder/cat.4001.jpg" 
sample_image = preprocess_image(sample_image_path)
prediction = clf.predict([sample_image])
if prediction[0] == 1:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")