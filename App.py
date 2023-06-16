# =================
#      IMPORT
# =================
from PIL import Image
from flask import Flask, request, jsonify, render_template
from keras.applications.resnet import preprocess_input
from keras.models import load_model
from numpy.linalg import norm  # Normalize
from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as np
import os
import pandas as pd
import pickle
import random


# =====================
#     FUNCTIONS
# =====================
def recommend(features, feature_list):
    neighbors = NearestNeighbors(
        n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result/norm(result)
    return normalized


# =================
# MODELS IMPORT
# =================
model_top_down = load_model("Models/model_top_down.h5")
model_extraction = load_model("Models/feature_extract_model.h5")

top_feature_list = np.array(pickle.load(
    open("Models/top_feature_extraction.pkl", "rb")))
bottom_feature_list = np.array(pickle.load(
    open("Models/bottom_feature_extraction.pkl", "rb")))

top_filenames = pickle.load(open('Models/top_directory.pkl', "rb"))
bottom_filenames = pickle.load(open("Models/bottom_directory.pkl", "rb"))

top_filenames_df = pickle.load(open('Models/top_directory_df.pkl', 'rb'))
bottom_filenames_df = pickle.load(open('Models/bottom_directory_df.pkl', 'rb'))

top_url = pickle.load(open("Models/top_url.pkl", "rb"))
bottom_url = pickle.load(open("Models/bottom_url.pkl", "rb"))

df = pd.read_csv("dataset.csv", index_col=False)


# ==================
# FLASK APP AND API
# ==================
app = Flask(__name__)


@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the Fashion Recommendation API'})


@app.route('/recommendation', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    file.save('Upload/' + file.filename)
    image_path = 'Upload/' + file.filename

    # Image preprocessing
    image = Image.open(image_path)
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the category
    prediction = model_top_down.predict(image)
    predicted_class = 'top' if prediction[0][0] > 0.5 else 'bottom'

    print(predicted_class)

    # Image recommendation
    # ====================
    responses = image_recommendation(predicted_class, image_path)

    print("Finished")
    os.remove(image_path)

    return responses


def image_recommendation(predicted_class, image_path):

    if predicted_class == 'top':
        features = extract_feature(image_path, model_extraction)
        indices = recommend(features, top_feature_list)

        # Recommendation file_path
        target_file = random.choice([top_filenames[indices[0][0]], top_filenames[indices[0][1]],
                                    top_filenames[indices[0][2]], top_filenames[indices[0][3]], top_filenames[indices[0][4]]])

        # Ambil setId dari rekomendasi
        filtered_df = df.loc[df['file_path'] == target_file]

        set_id = filtered_df["setId"]

        for set in set_id:
            set_id = set
            break

        print(set_id)

        # Teruskan setId ke dataframe bottom
        recommended_path = bottom_filenames_df[bottom_filenames_df['setId']
                                               == set_id]["file_path"]

        for path in recommended_path:
            recommended_path = path
            break

        print(recommended_path)

        # Recommend the bottom
        features = extract_feature(recommended_path, model_extraction)
        indices = recommend(features, bottom_feature_list)

        # Print the recommendation
        target_files = [
            bottom_filenames[indices[0][0]],
            bottom_filenames[indices[0][1]],
            bottom_filenames[indices[0][2]],
            bottom_filenames[indices[0][3]],
            bottom_filenames[indices[0][4]],
        ]

        price_output = []

        for file in target_files:
            filtered_df = bottom_filenames_df.loc[bottom_filenames_df['file_path'] == file]
            prices = filtered_df["price"].values
            if len(prices) > 0:
                price_output.append(prices[0])
            else:
                price_output.append('Unknown')

        target_link = [
            bottom_url[indices[0][0]],
            bottom_url[indices[0][1]],
            bottom_url[indices[0][2]],
            bottom_url[indices[0][3]],
            bottom_url[indices[0][4]],
        ]

    else:
        features = extract_feature(image_path, model_extraction)
        indices = recommend(features, bottom_feature_list)

        # Recommendation file_path
        target_file = random.choice([bottom_filenames[indices[0][0]], bottom_filenames[indices[0][1]],
                                    bottom_filenames[indices[0][2]], bottom_filenames[indices[0][3]], bottom_filenames[indices[0][4]]])

        # Ambil setId dari rekomendasi
        filtered_df = df.loc[df['file_path'] == target_file]

        set_id = filtered_df["setId"]

        for set in set_id:
            set_id = set
            break

        # Teruskan setId ke dataframe top
        recommended_path = top_filenames_df[top_filenames_df['setId']
                                            == set_id]["file_path"]

        for path in recommended_path:
            recommended_path = path
            break

        # Recommend the top
        features = extract_feature(recommended_path, model_extraction)
        indices = recommend(features, top_feature_list)

        # Print the recommendation
        target_files = [
            top_filenames[indices[0][0]],
            top_filenames[indices[0][1]],
            top_filenames[indices[0][2]],
            top_filenames[indices[0][3]],
            top_filenames[indices[0][4]],
        ]

        price_output = []

        for file in target_files:
            filtered_df = top_filenames_df.loc[top_filenames_df['file_path'] == file]
            prices = filtered_df["price"].values
            if len(prices) > 0:
                price_output.append(prices[0])
            else:
                price_output.append('Unknown')

        target_link = [
            top_url[indices[0][0]],
            top_url[indices[0][1]],
            top_url[indices[0][2]],
            top_url[indices[0][3]],
            top_url[indices[0][4]],
        ]

    new_price = []

    for price in price_output:
        # round price
        round_price = round(price)

        # modidy with idr to local currency
        idr_price = "IDR {:,}".format(round_price).replace(',', '.')
        new_price.append(idr_price)

    data = {
        'price_output': new_price,
        'target_link': target_link
    }

    responses = jsonify({
        'error': False,
        'data': data,
    })

    return responses


if __name__ == '__main__':
    app.run()
