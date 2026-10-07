#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
import numpy as np
from sklearn.model_selection import train_test_split

import cv2
import glob
import os
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense


# --------------------------------------------------------------------
# Configuration -- edit these paths (or set the matching env vars) to
# point at your own local copies of the dataset and outputs. Nothing
# here is committed with real user paths.
# --------------------------------------------------------------------
DATASET_ZIP_PATH = os.environ.get("DATASET_ZIP_PATH", "./data/archive.zip")
DATASET_EXTRACT_DIR = os.environ.get("DATASET_EXTRACT_DIR", "./data/extracted")
DATASET_BASE_PATH = os.environ.get("DATASET_BASE_PATH", "./data/glasses-noglasses")
MODEL_OUTPUT_PATH = os.environ.get("MODEL_OUTPUT_PATH", "./glasses_classifier.keras")


# In[2]:

if os.path.exists(DATASET_ZIP_PATH):
    with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATASET_EXTRACT_DIR)


# In[3]:


base_path = DATASET_BASE_PATH

train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "validate")
test_path = os.path.join(base_path, "test")


# In[4]:


print(os.listdir(train_path))


# In[5]:


print("Glasses:", len(os.listdir(os.path.join(train_path,"glasses"))))
print("No Glasses:", len(os.listdir(os.path.join(train_path,"noglasses"))))


# In[6]:


import os
import cv2
import numpy as np

data = []
labels = []

classes = ["glasses", "noglasses"]


def sobel_y_channel(img_float):
    gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely).astype(np.float32) / 255.0
    return np.expand_dims(sobely, axis=-1)


def add_sample(img_float, label):
    combined = np.concatenate((img_float, sobel_y_channel(img_float)), axis=-1)
    data.append(combined)
    labels.append(label)


def darken(img_float, gamma_range=(3.0, 7.0), factor_range=(0.07, 0.28), noise_range=(0.005, 0.02)):
    """Simulate a dark-room photo: crush shadows (gamma), dim overall exposure,
    keep a small random glow (e.g. a screen reflection) brighter than the rest,
    and add a touch of sensor noise."""
    h, w = img_float.shape[:2]
    gamma = np.random.uniform(*gamma_range)
    factor = np.random.uniform(*factor_range)
    shadowed = np.power(img_float, gamma) * factor

    glow_mask = np.random.random((8, 8)).astype(np.float32)
    glow_mask = cv2.resize(glow_mask, (w, h), interpolation=cv2.INTER_CUBIC)
    glow_mask = cv2.GaussianBlur(glow_mask, (0, 0), sigmaX=w / 6)
    glow_mask = (glow_mask - glow_mask.min()) / (glow_mask.max() - glow_mask.min() + 1e-6)
    glow_mask = glow_mask ** 3
    glow_boost = 1.0 + glow_mask[..., np.newaxis] * np.random.uniform(1.0, 2.5)

    dark = shadowed * glow_boost
    noise = np.random.normal(0, np.random.uniform(*noise_range), dark.shape).astype(np.float32)
    return np.clip(dark + noise, 0.0, 1.0)


for label, folder in enumerate(classes):

    folder_path = os.path.join(train_path, folder)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # Resize
        img = cv2.resize(img, (128, 128))

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize
        img = img.astype(np.float32) / 255.0

        # -------- Geometric augmentation: original + rotations/flips --------
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        angles = [45, 120, 150]

        variants = [img]
        for angle in angles:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            variants.append(rotated)
            variants.append(cv2.flip(rotated, 1))  # horizontal flip
            variants.append(cv2.flip(rotated, 0))  # vertical flip

        # -------- Emit each variant, plus a low-light counterpart --------
        for variant in variants:
            add_sample(variant, label)
            add_sample(darken(variant), label)

# Convert to NumPy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels)

print("Data Shape :", data.shape)
print("Labels Shape :", labels.shape)
print("Each image shape :", data[0].shape)


# In[7]:


# ==========================
# PREPROCESSING
# ==========================

X = np.array(data, dtype=np.float32)
y = np.array(labels)

# Add Channel Dimension
X = X.reshape(-1, 128, 128, 4)

print("X Shape:", X.shape)
print("y Shape:", y.shape)


# In[8]:


# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# In[9]:


# ==========================
# CNN MODEL
# ==========================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,4)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


# In[10]:


# ==========================
# COMPILE
# ==========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[11]:


# ==========================
# TRAIN
# ==========================
model.fit(
    X_train,
    y_train,
    epochs=10,
)

model.evaluate(X_test,y_test)


# In[23]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = (model.predict(X_test) > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=["Glasses", "No Glasses"])
disp.plot(cmap="Blues")
plt.show()


# In[56]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred,
                            target_names=["Glasses", "No Glasses"]))


# In[16]:


model.save(MODEL_OUTPUT_PATH)


# In[ ]:


# ==========================
# PREDICT ON A NEW IMAGE
# (uses the SAME preprocessing as training: RGB + normalized Sobel-Y channel)
# ==========================

def preprocess_image(img_bgr, size=128):
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobely = sobely.astype(np.float32) / 255.0
    sobely = np.expand_dims(sobely, axis=-1)

    combined = np.concatenate((img, sobely), axis=-1)  # shape (128,128,4)
    combined = np.expand_dims(combined, axis=0)         # shape (1,128,128,4)
    return combined, img

image_path = os.environ.get("TEST_IMAGE_PATH", "./data/test_image_1.jpg")

img_bgr = cv2.imread(image_path)
combined, display_img = preprocess_image(img_bgr)

prediction = model.predict(combined)

if prediction[0][0] > 0.5:
    print("Prediction : No Glasses")
else:
    print("Prediction : Glasses")

plt.imshow(display_img)
plt.axis("off")
plt.show()


# In[ ]:


# ==========================
# PREDICT ON A NEW IMAGE
# (uses the SAME preprocessing as training: RGB + normalized Sobel-Y channel)
# ==========================

def preprocess_image(img_bgr, size=128):
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobely = sobely.astype(np.float32) / 255.0
    sobely = np.expand_dims(sobely, axis=-1)

    combined = np.concatenate((img, sobely), axis=-1)  # shape (128,128,4)
    combined = np.expand_dims(combined, axis=0)         # shape (1,128,128,4)
    return combined, img

image_path = os.environ.get("TEST_IMAGE_PATH_2", "./data/test_image_2.jpg")

img_bgr = cv2.imread(image_path)
combined, display_img = preprocess_image(img_bgr)

prediction = model.predict(combined)

if prediction[0][0] > 0.5:
    print("Prediction : No Glasses")
else:
    print("Prediction : Glasses")

plt.imshow(display_img)
plt.axis("off")
plt.show()


# In[ ]:




