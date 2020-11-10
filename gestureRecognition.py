# import the opencv library
import operator

import cv2
import numpy as np
import tensorflow.keras

# define a video capture object
vid = cv2.VideoCapture(1)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    h1 = frame.shape[0]
    w1 = frame.shape[1]

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, h1, w1, 3), dtype=np.float32)

    # Replace this with the path to your image
    dimension = (w1, h1)
    resized_image = cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)
    image = resized_image

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    cv2.imshow("resized", resized_image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    #print(prediction)

    predictionDictionary = {
        "Zeigen" : prediction[0][0],
        "Hintergrund" : prediction[0][1],
        "Other" : prediction[0][2]
        }
    print(max(predictionDictionary.items(), key= operator.itemgetter(1))[0])

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
