#Import necessary libraries

from werkzeug.utils import secure_filename
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, Response

# keras is a open-source and neural network library which runs at the top of tensorflow

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # Preprocesses a tensor or Numpy array encoding a batch of images.s
from tensorflow.keras.preprocessing.image import img_to_array # converting the image to array.
from tensorflow.keras.models import load_model # Loading model which is trained and saved
import numpy as np # use array/matrix
import os # use to access file/folder in the machine
import cv2 # used for accessing the images for detection
from imutils.video import VideoStream # for video stream
import imutils


# path for face detection
prototxtPath = os.path.sep.join(['./face_detector','deploy.prototxt'])
caffePath = os.path.sep.join(['./face_detector','res10_300x300_ssd_iter_140000.caffemodel'])

# reading the face detection
faceNet = cv2.dnn.readNet(prototxtPath, caffePath)

# loading the trained model
maskNet = load_model('./face_mask_detector.model')


UPLOAD_FOLDER = './static/upload_image/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# function used to predict image
def predict_image(imagePath):

    # reading the image from specific path
    image = cv2.imread(imagePath)

    # getting heigth and width from a image
    (h, w ) = image.shape[ : 2]

    # preprocess the image
    blobImage = cv2.dnn.blobFromImage(image, 1.0,(300, 300),(104.0, 177.0, 123.0))

    # applying the blob image to the face detection
    faceNet.setInput(blobImage)
    detections = faceNet.forward()

    # looping over the detections
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # confidence of 50% - theroshold value
        if confidence > 0.5:

            # boundary box - geeting, x, y coordinates
            box = detections[0, 0, i, 3:7]*np.array([w,h,w,h])

            # starting and ending point of x, y coordinates
            (start_x, start_y, end_x, end_y) = box.astype('int')

            # ensuring the box falls within the frame
            (start_x, start_y) = (max(0,start_x), max(0,start_y))
            (end_x, end_y) = (min(w-1,end_x), min(h-1,end_y))

            # extract face ROi - Face ROI was a rectangular shape positioned automatically to cover the face
            face = image[start_y:end_y, start_x:end_x]

            # convert from BGR to RGB channel
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # resize it to the input size of the trained model which is 224x224
            face = cv2.resize(face, (224,224))

            # preprocessing it
            face  = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # prediting the value
            (mask,withoutmask) = maskNet.predict(face)[0]

            # creating the class label and color of the boundary box
            label = "Mask" if mask>withoutmask else 'No Mask'
            color = (0,255,0) if label == "Mask" else (0,0,255)

            # displaying the label and boundary box
            cv2.putText(image,label, (start_x, start_y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image,(start_x, start_y),(end_x,end_y), color, 2)

    # savinging the predicted image
    cv2.imwrite("./static/upload_image/Predicted_Output.png",image)


# function for predicting video
def predict_video(frame, faceNet, maskNet):

    # grab the dimensions of the frame and construct a blob
    (h,w) = frame.shape[:2]
    # preprocess the image
    blobImage = cv2.dnn.blobFromImage(frame, 1.0,(300, 300),(104.0, 177.0, 123.0))

    # appliying the blob image to the face detection
    faceNet.setInput(blobImage)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations and list of preditions
    faces = []
    locs = []
    preds = []

        # looping over the detections
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # confidence of 50% - theroshold value
        if confidence > 0.5:

            # boundary box - geeting, x, y coordinates
            box = detections[0, 0, i, 3:7]*np.array([w,h,w,h])

            # starting and ending point of x, y coordinates
            (start_x, start_y, end_x, end_y) = box.astype('int')

            # ensuring the box falls within the frame
            (start_x, start_y) = (max(0,start_x), max(0,start_y))
            (end_x, end_y) = (min(w-1,end_x), min(h-1,end_y))

            # extract face ROi - Face ROI was a rectangular shape positioned automatically to cover the face
            face = frame[start_y:end_y, start_x:end_x]

            # convert from BGR to RGB channel
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # resize it to the input size of the trained model which is 224x224
            face = cv2.resize(face, (224,224))

            # preprocessing it
            face  = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((start_x, start_y, end_x, end_y))

        # only make a predictions if atleast one face was detected.
        if len(faces) > 0:

            # converting the faces to array of type float32
            faces = np.array(faces, dtype='float32')

            # making the predicition
            preds = maskNet.predict(faces, batch_size=12)

        return (locs, preds)

#Initialize the Flask app
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # setting up the upload folder to the app
app.secret_key = "secret-key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# function for checking the upload image extension
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# function for displaying the home page
@app.route('/')
def index():
    return render_template('index.html')

# function redirect to the upload page
@app.route('/upload')
def upload():
  return render_template('upload.html')

# once the upload & predict button has been click it invoke the following function
@app.route('/upload', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are png, jpg, jpeg, gif')
		return redirect(request.url)

# displaying the uploaded image
@app.route('/upload/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='upload_image/' + filename))

# displaying the predicted image
@app.route('/upload/predict/<filename>')
def predict_image_display(filename):
	imagePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
	predict_image(imagePath)
	return redirect(url_for('static', filename='upload_image/Predicted_Output.png' ))

# generating the frame from the webcam
def generateFrames():
	vs = VideoStream(src=0).start()

	while True:

		# read the frame from the threaded videostream and resize it and have width of 400
		frame = vs.read()
		frame = imutils.resize(frame, width = 400)

		# detect faces in the frame and predicts the mask
		(locs, preds) = predict_video(frame, faceNet, maskNet)

		# looping over the detected face location and corresponding location
		for (box, pred) in zip(locs, preds):

			# starting and ending points of x, y coordinates
			(start_x, start_y, end_x, end_y) = box

			# prediting the values
			(mask,withoutmask) = pred

			# creating the class label and color of the boundary box
			label = "Mask" if mask>withoutmask else "No Mask"
			color = (0,255,0) if label == "Mask" else (0, 0, 255)

			# displaying the label and boundary box
			cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame,(start_x, start_y),(end_x, end_y), color, 2)
		ret, buffer = cv2.imencode('.jpg',frame)
		frame = buffer.tobytes()
		yield (b'--frame\r\n'
		b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		key = cv2.waitKey(1) & 0xFF

		if key == ord('q'):
			break
	vs.stop()
	cv2.destoryAllWindows()

# redirect to the live page
@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/webcam')
def webcam():
    return Response(generateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# running the flask app
if __name__ == "__main__":
    app.run(debug=True)