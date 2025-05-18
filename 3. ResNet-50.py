from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# Define ResNet-50 model
def create_resnet50(input_shape=(224, 224, 3), num_classes=1000):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=x)
    
    return model

# Create the model
resnet50_model = create_resnet50(input_shape=(224, 224, 3), num_classes=10)
resnet50_model.summary()
