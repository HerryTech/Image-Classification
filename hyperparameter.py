from sklearn.model_selection import ParameterGrid
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
def load_data():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # One-hot encode labels
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)
    
    return train_images, train_labels, test_images, test_labels

# Define grid search function
def grid_search(train_images, train_labels, test_images, test_labels):
    param_grid = {
        'conv_layers': [2, 3],
        'dense_units': [64, 128],
        'batch_size': [32, 64],
        'epochs': [5, 10]
    }
    
    grid = ParameterGrid(param_grid)
    best_model = None
    best_acc = 0

    # Loop through parameter grid
    for params in grid:
        print(f"Training with parameters: {params}")
        
        # Build the model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            *(layers.Conv2D(64, (3, 3), activation='relu') for _ in range(params['conv_layers'] - 1)),
            layers.Flatten(),
            layers.Dense(params['dense_units'], activation='relu'),
            layers.Dense(10, activation='softmax')  # CIFAR-10 has 10 classes
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(
            train_images, train_labels,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(test_images, test_labels),
            verbose=1
        )
        
        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f"Accuracy: {test_acc * 100:.2f}%")
        
        # Track the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model

    print(f"Best accuracy achieved: {best_acc * 100:.2f}%")
    return best_model

# Load data
train_images, train_labels, test_images, test_labels = load_data()

# Run grid search
optimized_model = grid_search(train_images, train_labels, test_images, test_labels)
