# import the necessary packages

# keras is a open-source and neural network library which runs at the top of tensorflow

from tensorflow.keras.applications import MobileNetV2 # MobileNetV2 as a model - ligthweigth model 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # Preprocesses a tensor or Numpy array encoding a batch of images.s

from tensorflow.keras.layers import AveragePooling2D # importing the Average pooling operation for spatial data.
from tensorflow.keras.layers import Dropout # which randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting.
from tensorflow.keras.layers import Flatten # flatten which is used to Flattens the input
from tensorflow.keras.layers import Dense # Dense layer which is just regular densely-connected NN layer.
from tensorflow.keras.layers import Input # used to instantiate a Keras tensor.
from tensorflow.keras.models import Model # Importing the model which is used to Model groups layers into an object with training and inference features
from tensorflow.keras.optimizers import Adam # Optimizer that implements the Adam algorithm.

from tensorflow.keras.preprocessing.image import ImageDataGenerator # Generate batches of tensor image data with real-time data augmentation.
from tensorflow.keras.preprocessing.image import img_to_array #  Converts a  Image instance to a Numpy array.
from tensorflow.keras.preprocessing.image import load_img # Loads an image into PIL format.
from tensorflow.keras.utils import to_categorical # Converts a class vector (integers) to binary class matrix.

from sklearn.preprocessing import LabelBinarizer # convert multi-class labels to binary labels (belong or does not belong to the class).
from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets
from sklearn.metrics import classification_report # Build a text report showing the main classification metrics.

from imutils import paths # The function returns all the names of the files in the directory path supplied as argument to - accessing the path
import matplotlib.pyplot as plt # visualizing in form of plot graph
import numpy as np # use array/matrix
import os # use to access file/folder in the machine


# specifing the dataset location

dataset = r'E:\my learn\AI - DS\DL\mask_detection\dataset'
imagePath = list(paths.list_images(dataset))

# creating variable to store images and labels
image_data = []
labels = []

for i in imagePath:
  label = i.split(os.path.sep)[-2] # spliting with folder name
  labels.append(label)

  # resizing all the images to specific size
  image = load_img(i, target_size=(224,224))

  # converting the image to an array based on pixel values - ranges from 0 to 255
  image = img_to_array(image)

  # preprocessing the image according to the mobilenet - normalize by scaling the value
  image = preprocess_input(image)

  # adding up the prepocessed image data
  image_data.append(image)

# convert the image_data and labels to NumPy arrays
image_data = np.array(image_data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels) # fitting the categorial labels
labels = to_categorical(labels) # converting the string label into interger labels

# partition the data into training and testing splits using 80% of the data for training and the remaining 20% for testing
# test_size = is used to partition the test data based on the specific precentage from the whole dataset
# random_state = used to partition the test and train data based on random integers
# stratify = make proper arrangement for partition - same distribution of classes on both sets.

train_x, test_x, train_y, test_y = train_test_split(image_data, labels, test_size=0.20, random_state=10, stratify=labels )

# data augmentation for generating more dataset/image

# rescale = rescale/normalizing the image based on pixcel value
# zoom_range = zooming any part of the image with a specified range, value in percentage.
# rotation_range = changing the rotation of the image, value is of degree
# flip = making a flip of an image either horizontal or vertical
# shift_range = shifting or moveing the image either by width or height
# shear_range = randomly applying shearing transformations
# fill_mode = replaces the empty area with the nearest/constant pixel values

data_aug = ImageDataGenerator(rotation_range=20,
                              zoom_range=0.15,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.15,
                              horizontal_flip=True,
                              vertical_flip = True,
                              fill_mode='nearest')

# load the MobileNetV2 network, ensuring the head FullyConnected layer sets are left off
# weights = transforms input data within the network's hidden layers.
# imagenet = ImageNet dataset is a very large collection of human annotated photographs
# include_top = including the dense/ fully connected layer.
# input_tensor = size of the input data.

baseModel = MobileNetV2(weights='imagenet',
                        include_top=False,
                        input_tensor=Input(shape=(224,224,3)))

# getting the summary of the model
baseModel.summary()


# construct the head of the model that will be placed on top of the the base model
headModel = baseModel.output

# using the Average polling layer
# avgpooling = takes the average value
# pooling size = size of the pooling
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)

# Adding up the faltten layer which is used to flatten a 2d matrix to a single vector value/features
headModel = Flatten(name="flatten")(headModel)

# making a fully connected layer and making a activation by using the RELU = max(value,0)
headModel = Dense(128, activation="relu")(headModel)

# Adding up the Dropout layer because of overfeeding the data
headModel = Dropout(0.5)(headModel)

# making a fully connected layer and making a activation by using the softmax = exp(value) / sum values in list exp(values), probability function
headModel = Dense(2, activation="softmax")(headModel)

# making a complete model by placing the head FullyConnected model on top of the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# model summary
model.summary()

# loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

learning_rate = 0.001 # learning rate at which model learns
Epochs = 20 # No. of iternation a model perform
BS = 12 # No. of image_data per batch

# optimising using adam algorithm for achieving the global minima
# lr = learning rate which determine the updating step influences the current value of the weight
# decay = causes the weights to exponentially decay to zero, if no other update is scheduled.

optim = Adam(lr=learning_rate, decay=learning_rate / Epochs)

# compling the model
# loss = the loss value
# matric = perfomance

model.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])


# train the head of the network
History = model.fit(
	data_aug.flow(train_x, train_y, batch_size=BS),
	steps_per_epoch=len(train_x) // BS,
	validation_data=(test_x, test_y),
	validation_steps=len(test_x) // BS,
	epochs=Epochs)

# saving the model
model.save('.\model\face_mask_detector.model')
# model.save("/content/face_mask_detector", save_format='h5')

# evaluating the model prediction
predict = model.predict(test_x, batch_size=BS)

# checking which output have maximum probability
predict = np.argmax(predict, axis=1)

# show a nicely formatted classification report
print(classification_report(test_y.argmax(axis=1), predict ,target_names=lb.classes_))

# plot the training loss and accuracy
N = Epochs
# using the ggplot to plot the data
plt.style.use("ggplot")
plt.figure()

# ploting the value for loss, accuracy, val_loss and val_accuracy
plt.plot(np.arange(0, N), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), History.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), History.history["val_accuracy"], label="val_acc")

# giving out the title for the graph
plt.title("Training and Validation - Loss and Accuracy")

# labeling x-axis and y -axis
plt.xlabel("# Epoch")
plt.ylabel("Loss/Accuracy")

# Displaying the label hints/scale
plt.legend()

# saving the ploted image.
plt.savefig("Training_accuracy.png")
