# Wave Annotation Bot 🌊🤖
This project trains a deep learning model to predict the quality of a wave from an image. It uses a Convolutional Neural Network (CNN) based on the VGG16 architecture to learn from a dataset of pre-annotated wave images and score new ones. The goal is to create a tool that can automatically rate surf quality.

## 🚀 Getting Started
Prerequisites
Python 3.8+

Git

A dataset of wave images. Each image file must be correctly named with a surf quality score.

Installation
Clone the repository:

Bash

git clone https://github.com/your-username/wave-annotation-bot.git
cd wave-annotation-bot
Install the required Python packages:

Bash

pip install tensorflow numpy scikit-learn
## 📊 Dataset Preparation
The model learns from your data. The more high-quality, diverse, and correctly named images you have, the better the model will perform.

Create a folder named wave_dataset in the project's root directory.

Add your wave images to this folder.

Name your image files using the following convention to include the surf quality score:
wave_score_XX_id_XXX.png

XX: A two-digit score from 01 to 10 (e.g., 05 for a score of 5).

XXX: A unique three-digit ID for the image (e.g., 123).

Example filenames:

wave_score_08_id_001.jpg

wave_score_02_id_002.png

## 💻 Usage
Once your dataset is ready, the script will automatically load the data, train the model, and save it.

Run the main script:

Bash

python main.py
The script will:

Scan the wave_dataset directory for images.

Load and preprocess the images and their scores.

Split the data into training and testing sets.

Train the neural network using transfer learning with the VGG16 model.

Evaluate the model's performance on the test set, reporting the Mean Absolute Error (MAE). This metric tells you, on average, how far off the model's predictions are from the actual scores.

Save the trained model as wave_quality_model.h5.

After the model is saved, you can use it to predict the score of a new, unseen wave image.
