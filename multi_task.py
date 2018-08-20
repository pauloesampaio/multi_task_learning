from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras.models import Model
from keras.layers import Dense, Input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import to_categorical
import json
import numpy as np

# Batch size really depends on your GPU memory and model architecture
BATCH_SIZE = 32

# Training set is a database of the format:
#{file_id_1: {task_1: label, task_2: label, task_3: label}, 
# file_id_2: {task_1: label, task_2: label, task_3: label},
# ...
# file_id_3: {task_1: label, task_2: label, task_3: label},
#}
# I recommend having it as a json file that could easily be 
# loaded with:
# with open("./training_set.json","r") as f:
#     TRAINING_SET = json.load(f)

# Lable dict is a dictionary that, for each task,
# links each label to a numerical index. Something like:
# LABEL_DICT = {
#    "task_1": {'label_0': 0, 'label_1': 1, 'label_2': 2,
#               'label_3': 3, ... ,'label_19': 19},
#    "task_2": {"label_0": 0, "label_1": 1, "label_2": 2},
#    "task_3": {"label_0": 0, "label_1": 1, "label_2": 2},
#}


# Preprocess image is part of the pre-trained architecture we'll be working with
def preprocess_image(current_image):
    current_image = preprocess_input(current_image.reshape((1,)+current_image.shape)) 
    return current_image

# This will give us the file names of the images of each batch of X, 
# so we can get the labels from our training set database
class FilesIterator(Iterator):
    # This will receive a generator and get all the information from it (mainly filenames, batch size, and random seed)
    def __init__(self, generator):
        self.file_list = generator.filenames
        self.batch_size = generator.batch_size
        self.shuffle = generator.shuffle
        self.seed = generator.seed
        self.n = len(generator.filenames)
        super(FilesIterator, self).__init__(self.n, self.batch_size, self.shuffle, self.seed)
        
    # This will return the filenames given an index
    def _get_batches_of_transformed_samples(self, index_array):
        current_files = [self.file_list[w] for w in index_array]
        return current_files

    # This will get indexes and call the function that return filenames
    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)
      

# Loading Resnet with imaginet weights and without the classification layer    
model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_tensor=Input(shape=(224,224,3)))

# Adding as many custom classification layer with as many classes needed
task_1_clf = Dense(20, activation="softmax", name="task_1_clf")(model.output)
task_2_clf = Dense(3, activation="softmax", name="task_2_clf")(model.output)
task_3_clf = Dense(2, activation="softmax", name="task_3_clf")(model.output)
model = Model(inputs = model.input, outputs = [task_1_clf, task_2_clf, task_3_clf])


# Starting the image data generator for the training set
train_image_datagen = ImageDataGenerator(rotation_range=15, 
                                         width_shift_range=.2, 
                                         height_shift_range=.2,
                                         horizontal_flip=True, 
                                         preprocessing_function=preprocess_image)

# Instructing it to run on the folders with the data. 
# The batch size depends on how much memory your GPU have       
# Notice the None on the class_mode parameter
train_image_generator = train_image_datagen.flow_from_directory('./data/train/',  
                                                                target_size=(224, 224), 
                                                                batch_size=BATCH_SIZE, 
                                                                class_mode=None, 
                                                                shuffle=True, 
                                                                seed=12345) 

# Running our files generator on the train image generator
train_files_generator = FilesIterator(train_image_generator)


# Same thing for test data
test_image_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
test_image_generator = test_image_datagen.flow_from_directory('./data_old/val/',  
                                                               target_size=(224, 224), 
                                                               batch_size=BATCH_SIZE, 
                                                               class_mode=None, 
                                                               shuffle=False, 
                                                               seed=12345) 
test_files_generator = FilesIterator(test_image_generator)


# Having the image generator and the file name generator,
# use both to get batches of (X,y)
def custom_generate_batches(IMAGE_GENERATOR, FILES_GENERATOR):
    # Task list
    TASKS = ["task_1", "task_2", "task_3"]
    while True:
        # Get X from the image generator
        X = IMAGE_GENERATOR.next()
        
        # Get file names from the files generator
        X_files = FILES_GENERATOR.next()
        
        # For each one of the tasks, get the label from the training set 
        # database. We'll also get the label index from our label dictionary, 
        # and pass it to a one-hot vector using the "to_categorical" 
        # function.
        Y = []
        for task in TASKS:
            current_labels = [TRAINGING_SET[file_name][task] for file_name in X_files]
            encoded_labels = np.asarray([LABEL_DICT[task][label] for label in current_labels], dtype=K.floatx())
            Y.append(to_categorical(encoded_labels, len(LABEL_DICT[task])))
        yield X, Y


# Set the optimisation algorithm 
adam = Adam(lr=0.001)

# Compiling the model
model.compile(loss="categorical_crossentropy",
              optimizer=adam, metrics=["accuracy"])

# Fit the model based on the data generators we saw before
# Running it for 10 epochs
model.fit_generator(
    generator=custom_generate_batches(train_image_generator, train_files_generator),
    steps_per_epoch=train_generator.n/BATCH_SIZE,
    validation_data=custom_generate_batches(test_image_generator, test_files_generator),
    validation_steps=test_generator.n/BATCH_SIZE,
    epochs=10)
