# This repository is used for ML learning path. It contains the following branches:

| Branch| Description|
| --- | --- |
| [main](https://github.com/Fashionism-Bangkit-Capstone/Fashionism-Machine-Learning.git)  | This branch contains the main documentation for the ML learning path. |
| [model-development](https://github.com/Fashionism-Bangkit-Capstone/Fashionism-Machine-Learning/tree/model-development) | This branch contains the model development process. |
| [model-deployment](https://github.com/Fashionism-Bangkit-Capstone/Fashionism-Machine-Learning/tree/model-deployment) | This branch contains the model deployment process. |

# TOP_BOTTOM_MODELLING Documentation
![Modelling](Images/Fashionism%20Roadmap%20(1).png)
This code performs the task of constructing a dataset, downloading images, and training a model for classifying top and bottom clothing items. The documentation provides an overview of the code structure and its functionality.

## Code Overview
The code is written in Python and runs in a Jupyter Notebook environment. It consists of several sections, each performing a specific task. Here is an overview of each section:

### Importing Dependencies
The code begins by importing the necessary dependencies, including libraries such as pandas, requests, zipfile, and concurrent.futures. It also imports specific modules from these libraries and sets up additional configurations.

### CSV Extracting
In this section, the code uploads and extracts CSV files containing image and style data. It uses the `files.upload()` function to upload the files and the `!unzip` command to extract them. After extraction, the original zip files are removed.

### Dataset Loading and Processing
This section focuses on loading and processing the dataset for training the model. It reads the CSV files into Pandas DataFrames and merges them based on the "filename" column. It filters the data to include only "Apparel" items and changes the "subCategory" values to "Top" and "Bottom" accordingly. The code then balances the dataset by sampling an equal number of rows for each category. Finally, it shuffles the rows of the balanced DataFrame.

### Image Downloading
In this section, the code downloads the images corresponding to the URLs in the dataset. It defines a function to download an image given a URL and local directory. It creates a directory called "dataset" and uses multithreading to download images concurrently. The progress of the download tasks is tracked, and the completed count is displayed.

### Modeling
![VGG Structure](Images/1_NNifzsJ7tD2kAfBXt3AzEg.webp)
This section focuses on training the classification model for top and bottom clothing items. It splits the dataset into training and testing sets. It also creates an ImageDataGenerator for data augmentation and preprocessing. The VGG16 pre-trained model is loaded and frozen as the base model. The code then constructs the model architecture by adding additional layers on top of the base model. The model is compiled with appropriate optimizer and loss functions. Early stopping is implemented to prevent overfitting during training. The model is trained on the training set and evaluated on the test set. Finally, the trained model is saved as "top_down_new_model.h5".

## Usage and Results
To use the code, it needs to be executed in a Jupyter Notebook environment. The code performs the following tasks:
1. Extracts CSV files containing image and style data.
2. Balances and preprocesses the dataset.
3. Downloads the corresponding images from URLs.
4. Trains a model using the VGG16 architecture to classify top and bottom clothing items.
5. Evaluates the model's performance on the test set.
6. Saves the trained model for future use.

Throughout the code execution, progress updates and relevant information are displayed.

The final trained model ("top_down_new_model.h5") can be used for predicting the category (top or bottom) of new clothing images.

Note: It's important to review and understand the code before running it to ensure it fits your specific requirements and dataset structure.

This concludes the documentation for the provided code.

# Flask Deployment Documentation (App.py)

The full code can be seen on [This branch](https://github.com/Fashionism-Bangkit-Capstone/Fashionism-Machine-Learning/tree/model-deployment).

## Importing Libraries and Modules

The code begins by importing the necessary libraries and modules for the application. The imported modules include:

- `Image` from the PIL library: Used for image processing and manipulation.
- `Flask`, `request`, `jsonify`, and `render_template` from the flask library: Used for creating a Flask application, handling HTTP requests, and rendering HTML templates.
- `preprocess_input` from `keras.applications.resnet`: Used for preprocessing input images for the ResNet model.
- `load_model` from `keras.models`: Used for loading pre-trained models.
- `norm` from `numpy.linalg`: Used for normalizing feature vectors.
- `NearestNeighbors` from `sklearn.neighbors`: Used for performing nearest neighbor search.
- `cv2` and `numpy` (imported as `np`): Used for image processing and numerical operations.
- `os` and `pickle`: Used for file and data serialization operations.
- `pd` (imported as `pandas`): Used for data manipulation and analysis.
- `random`: Used for generating random choices.

## Functions

### `recommend(features, feature_list)`
This function performs nearest neighbor search on a given feature vector using a precomputed feature list.
- `features`: A feature vector for which neighbors need to be found.
- `feature_list`: A precomputed feature list used for neighbor search.
Returns the indices of the nearest neighbors.

### `extract_feature(img_path, model)`
This function extracts features from an image using a pre-trained model.
- `img_path`: The path to the image file.
- `model`: The pre-trained model used for feature extraction.
Returns the normalized feature vector of the image.

## Model and Data Loading

The code proceeds to load pre-trained models and data needed for image recommendation. The loaded items include:
- `model_top_down`: A pre-trained model for predicting whether an image is a top or bottom.
- `model_extraction`: A pre-trained model for feature extraction.
- `top_feature_list` and `bottom_feature_list`: Precomputed feature lists for top and bottom clothing items.
- `top_filenames` and `bottom_filenames`: Lists of file names corresponding to top and bottom clothing items.
- `top_filenames_df` and `bottom_filenames_df`: DataFrames containing additional information about top and bottom clothing items.
- `top_url` and `bottom_url`: Lists of URLs corresponding to top and bottom clothing items.
- `df`: A pandas DataFrame containing additional dataset information.

## Flask App and API

The code initializes a Flask application and defines the following routes and functions:

### Route: `/`
- Function: `home()`
- Method: GET
- Description: This route renders the 'index.html' template and serves it as the home page of the web application.

### Route: `/upload`
- Function: `upload_file()`
- Method: POST
- Description: This route handles the file upload functionality. It expects a file to be uploaded with the name 'file'. It saves the uploaded file, preprocesses the image, predicts its category (top or bottom), and performs image recommendation based on the predicted category.

### Function: `image_recommendation(predicted_class, image_path)`
- Description: This function performs image recommendation based on the predicted category and the image path.
- `predicted_class`: The predicted category of the uploaded image ('top' or 'bottom').
- `image_path`: The path to the uploaded image file.
- Returns: A JSON response containing the recommended items' price and corresponding links.

The Flask application is then run if the script is executed directly.

---

This code implements a Flask application that allows users to upload an image of a clothing item and receive recommendations for

 complementary clothing items (top-bottom or bottom-top combinations). The recommendations are based on pre-trained models for predicting clothing categories and extracting features from images. The application uses nearest neighbor search to find the most similar items in the feature space and provides their prices and links for further exploration.