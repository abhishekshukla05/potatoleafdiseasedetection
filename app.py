# import os
# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
# from skimage import io
# from tensorflow.keras.preprocessing import image


# # Flask utils
# from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# # Define a flask app
# app = Flask(__name__)

# # Model saved with Keras model.save()

# # You can also use pretrained model from Keras
# # Check https://keras.io/applications/

# model =tf.keras.models.load_model('PlantDNet.h5',compile=False)
# print('Model loaded. Check http://127.0.0.1:5000/')


# def model_predict(img_path, model):
#     img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
#     show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = np.array(x, 'float32')
#     x /= 255
#     preds = model.predict(x)
#     return preds


# @app.route('/', methods=['GET'])
# def index():
#     # Main page
#     return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         preds = model_predict(file_path, model)
#         print(preds[0])

#         # x = x.reshape([64, 64]);
#         disease_class = ['Potato___Early_blight','Potato___Late_blight', 'Potato___healthy']
#         a = preds[0]
#         ind=np.argmax(a)
#         print('Prediction:', disease_class[ind])
#         result=disease_class[ind]
#         return result
#     return None


# if __name__ == '__main__':
#     app.run(port=5002, debug=True)

#     # Serve the app with gevent
#     # http_server = WSGIServer(('', 5000), app)
#     # http_server.serve_forever()
#     app.run()
# # from flask import Flask, render_template, request
# # import tensorflow as tf
# # import numpy as np
# # from PIL import Image

# # # Load the trained model
# # model = tf.keras.models.load_model('PotatoDiseaseClassifierModel.h5')

# # # Create a Flask application
# # app = Flask(__name__)

# # # Define the home route
# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # # Define the predict route for handling image uploads and making predictions
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     # Get the uploaded image file from the request
# #     image = request.files['file']
    
# #     # Load and preprocess the image
# #     img = Image.open(image)
# #     img = img.resize((256, 256))  # Resize the image to match the input size of the model
# #     img = np.array(img) / 255.0  # Normalize pixel values to the range [0, 1]
# #     img = np.expand_dims(img, axis=0)  # Add a batch dimension
    
# #     # Make the prediction using the loaded model
# #     prediction = model.predict(img)
    
# #     # Get the predicted class label
# #     labels = ['Healthy', 'Early Blight', 'Late Blight']
# #     predicted_class = labels[np.argmax(prediction)]
    
# #     # Render the results template with the predicted class
# #     return render_template('results.html', predicted_class=predicted_class)

# # # Run the Flask application
# # if __name__ == '__main__':
# #     app.run(debug=True)
# import streamlit as st
# from PIL import Image
# import tensorflow as tf
# import numpy as np

# # Load the trained model
# model = tf.keras.models.load_model('C:/Users/chinmay/Videos/codes/python/Plant-Disease-Diagnosis-Flask-master/PlantDNet.h5')

# # Define the class labels

# class_labels = ['Potato___Early_blight','Potato___Late_blight', 'Potato___healthy']
# # Set up the Streamlit app
# st.title('Potato Health Classification')

# # Create a file uploader
# uploaded_file = st.file_uploader('Upload an image of a potato', type=['jpg', 'jpeg', 'png'])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     # Preprocess the image
#     # resized_image = image.resize((256, 256))  # Adjust the size as per your model requirements
#     # normalized_image = resized_image / 255.0
#     # input_image = tf.expand_dims(normalized_image, axis=0)
#     resized_image = image.resize((64, 64))  # Adjust the size as per your model requirements
#     normalized_image = np.array(resized_image) / 255.0
#     input_image = tf.expand_dims(normalized_image, axis=0)
    
#     # Make predictions
#     predictions = model.predict(input_image)
#     predicted_class = class_labels[predictions.argmax()]
#     confidence = predictions.max() * 100
    
#     # Display the predicted class and confidence
#     st.subheader('Prediction:')
#     st.write(f'Class: {predicted_class}')
#     st.write(f'Confidence: {confidence:.2f}%')

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model =tf.keras.models.load_model('PlantDNet.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);
        disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                         'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                         'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                         'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                         'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
        a = preds[0]
        ind=np.argmax(a)
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        return result
    return None

if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()

