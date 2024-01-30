import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def predict():
    # Replace this with the path to your image
    image = Image.open('static/images/test_image.jpg')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    idx = np.argmax(prediction)

    if idx == 0:
        return "Soil type: Black Soil, Suitable Crops:Mushroom"
    elif idx == 1:
        return "Soil type: Desert Soil, Suitable Crops:Pea"
    elif idx == 2:
        return "Soil type: Laterrite Soil, Suitable Crops:Wheat"
    elif idx == 3:
        return "Soil type: Mountain Soil, Suitable Crops:Rice"
    elif idx == 4:
        return "Soil type: Red Soil, Suitable Crops:Jowar"   
    else:
        return "Unknown Soil Type"
