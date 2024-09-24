import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import tensorflow as tf
import certifi
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.losses import MeanSquaredError

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    obj = root.find('object')
    bndbox = obj.find('bndbox')
    xmin = float(bndbox.find('xmin').text)
    ymin = float(bndbox.find('ymin').text)
    xmax = float(bndbox.find('xmax').text)
    ymax = float(bndbox.find('ymax').text)
    return [xmin, ymin, xmax, ymax]

def load_dataset(images_dir, xml_dir):
    images = []
    bboxes = []
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(images_dir, filename)
            xml_path = os.path.join(xml_dir, filename.replace('.jpg', '.xml'))
            if os.path.exists(xml_path):
                try:
                    bbox = parse_xml(xml_path)
                    images.append(img_path)
                    bboxes.append(bbox)
                except Exception as e:
                    print(f"Error parsing {xml_path}: {e}")
            else:
                print(f"XML file not found for {img_path}")
    return images, bboxes

def preprocess_images(images, bboxes, img_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    dataframe = pd.DataFrame({'filename': images, 'bbox': bboxes})
    
    # Convert bbox columns to float32
    bbox_array = np.array(bboxes, dtype=np.float32)
    dataframe = pd.DataFrame({
        'filename': images,
        'xmin': bbox_array[:, 0],
        'ymin': bbox_array[:, 1],
        'xmax': bbox_array[:, 2],
        'ymax': bbox_array[:, 3]
    })
    
    # Debug print to check the DataFrame structure
    print("DataFrame head:")
    print(dataframe.head())
    print("DataFrame info:")
    print(dataframe.info())
    
    train_generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='filename',
        y_col=['xmin', 'ymin', 'xmax', 'ymax'],
        target_size=img_size,
        batch_size=32,
        class_mode='raw',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='filename',
        y_col=['xmin', 'ymin', 'xmax', 'ymax'],
        target_size=img_size,
        batch_size=32,
        class_mode='raw',
        subset='validation'
    )
    
    return train_generator, validation_generator

def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4)(x)  # 4 outputs for bounding box coordinates
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['accuracy'])
    return model

# Set SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load dataset
images_dir = '/Users/sanjanakusupudi/Downloads/archive/Mobile_image/Mobile_image'
xml_dir = '/Users/sanjanakusupudi/Downloads/archive/Annotations/Annotations'
images, bboxes = load_dataset(images_dir, xml_dir)

# Debug prints to check loaded data
print(f"Loaded {len(images)} images and {len(bboxes)} bounding boxes")

# Preprocess images
train_generator, validation_generator = preprocess_images(images, bboxes)

# Build model
model = build_model()

# Train model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy:.2f}')

# Function to predict bounding box for a new image
def predict_bounding_box(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction[0]

# Example usage
image_path = '/Users/sanjanakusupudi/Downloads/archive/Mobile_image/Mobile_image/Datacluster Labs Phone Dataset (77).jpg'
bbox = predict_bounding_box(image_path)
print(f'Predicted bounding box: {bbox}')
