from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.optimizers import Adam

# Batch size really depends on your GPU memory and model architecture
BATCH_SIZE = 32

# Preprocess image is part of the pre-trained architecture we'll be working with
def preprocess_image(current_image):
    current_image = preprocess_input(current_image.reshape((1,)+current_image.shape)) 
    return current_image


# Loading Resnet with imaginet weights and without the classification layer
model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_tensor=Input(shape=(224,224,3)))

# Adding a custom classification layer with 20 classes
task_1_clf = Dense(20, activation="softmax", name="task_1_clf")(model.output)
model = Model(inputs = model.input, outputs = task_1_clf)

# Starting the image data generator for the training set
train_datagen = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=.20,
                                   height_shift_range=.20,
                                   horizontal_flip=True,
                                   preprocessing_function=preprocess_image)

# Instructing it to run on the folders with the data. 
# The batch size depends on how much memory your GPU have                   
train_generator = train_datagen.flow_from_directory('./data/train/', 
                                                    target_size=(224, 224),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="categorical",
                                                    shuffle=True,
                                                    seed=12345) 


# Same thing for test set
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
test_generator = train_datagen.flow_from_directory('./data/val/',  
                                                   target_size=(224, 224), 
                                                   batch_size=BATCH_SIZE, 
                                                   class_mode="categorical", 
                                                   shuffle=False, 
                                                   seed=12345) 

# Set the optimisation algorithm 
adam = Adam(lr=0.001)

# Compiling the model
model.compile(loss="categorical_crossentropy",
              optimizer=adam,
              metrics=["accuracy"])


# Fit the model based on the data generators we saw before
# Running it for 10 epochs
model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.n/BATCH_SIZE,
                    validation_data=test_generator,
                    validation_steps=test_generator.n/BATCH_SIZE,
                    epochs=10)
