# This repository is used for ML learning path. It contains the following branches:

| Branch| Description|
| --- | --- |
| [main](https://github.com/Fashionism-Bangkit-Capstone/Fashionism-Machine-Learning.git)  | This branch contains the main documentation for the ML learning path. |
| [model-development](https://github.com/Fashionism-Bangkit-Capstone/Fashionism-Machine-Learning/tree/model-development) | This branch contains the model development process. |
| [model-deployment](https://github.com/Fashionism-Bangkit-Capstone/Fashionism-Machine-Learning/tree/model-deployment) | This branch contains the model deployment process. |

# TOP_BOTTOM_MODELLING Documentation

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