from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import os
from PIL import Image


app = Flask(__name__)

model = keras.models.load_model('ResNet_Model.h5')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get the file from the HTTP request
    file = request.files['file']

    # save the file to the uploads folder
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # convert image to RGB if it has an alpha channel
    img = Image.open(filepath)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
        img.save(filepath)

    # load the image and preprocess it
    img = Image.open(filepath)
    img = img.resize((256, 256))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    labels = ['Bear', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar', 'Cattle', 'Centipede', 'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer', 'Duck', 'Eagle', 'Elephant', 'Fish', 'Fox', 'Frog', 'Giraffe', 'Goat', 'Goldfish', 'Goose', 'Hamster', 'Harbor seal', 'Hedgehog', 'Hippopotamus', 'Horse', 'Jaguar', 'Jellyfish', 'Kangaroo', 'Koala', 'Ladybug', 'Leopard', 'Lion', 'Lizard', 'Lynx', 'Magpie', 'Monkey', 'Moths and butterflies', 'Mouse', 'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda', 'Parrot', 'Penguin', 'Pig', 'Polar bear', 'Rabbit', 'Raccoon', 'Raven', 'Red panda', 'Rhinoceros', 'Scorpion', 'Sea lion', 'Sea turtle', 'Seahorse', 'Shark', 'Sheep', 'Shrimp', 'Snail', 'Snake', 'Sparrow', 'Spider', 'Squid', 'Squirrel', 'Starfish', 'Swan', 'Tick', 'Tiger', 'Tortoise', 'Turkey', 'Turtle', 'Whale', 'Woodpecker', 'Worm', 'Zebra']


    # make a prediction on the image
    prediction = model.predict(x)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = labels[predicted_class_index]

    # return the predicted animal name
    return f"<h1 style='color: green;'>The uploaded image contains a: {predicted_class_label}</h1>"


if __name__ == '__main__':
    app.run(debug=True)
