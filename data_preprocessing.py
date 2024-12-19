import tensorflow as tf
import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-100 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

cifar100_dir = tf.keras.utils.get_file(
    "cifar-100-python",
    origin="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    untar=True
)

# Get class names
for root, dirs, files in os.walk(cifar100_dir):
    print(f"Checking in: {root}")
    for file in files:
        print(file)
        # Look for the 'meta' file
        if file == 'meta':
            meta_file_path = os.path.join(root, file)
            with open(meta_file_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                class_names = [name.decode('utf-8') for name in data[b'fine_label_names']]
            print("Class names:", class_names)
            print("Number of classes:", len(class_names))
            break

# Define class names manually (if 'meta' file isn't found or for reference)
class_names = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
    "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
    "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
    "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower",
    "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

# Normalize the dataset
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Define the output directory for augmented images
output_dir = "aug_dataset"
os.makedirs(output_dir, exist_ok=True)

# Augmentation setup
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Create a subfolder for each class in the output directory using class names
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# Generate and save augmented images
num_augmented_images_per_class = 500

for class_index, class_name in enumerate(class_names):  # Loop over each class using index and name
    # Select all images belonging to the current class
    class_images = train_images[train_labels.flatten() == class_index]
    class_dir = os.path.join(output_dir, class_name)
    
    # Use the class name as the prefix to save images with the class name
    aug_iter = datagen.flow(
        class_images,
        batch_size=1,
        save_to_dir=class_dir,
        save_prefix=class_name,  # Use class name as prefix
        save_format='png'
    )

    # Generate and save images
    for img_num in range(num_augmented_images_per_class):
        next(aug_iter)  # Generate and save one augmented image per iteration

print(f"Augmented images saved in {output_dir}")
