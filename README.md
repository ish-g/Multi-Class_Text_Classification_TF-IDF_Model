# Multi-Class_Text_Classification_TF-IDF_Model

📝 Sentiment Analysis Web App with BaggingClassifier and SVC
Welcome to the Sentiment Analysis Web App project! This repository contains a web application built using Streamlit that performs sentiment analysis on user-input text using a BaggingClassifier with a Support Vector Classifier (SVC) as the base estimator. The app is designed to classify the sentiment of text as positive or negative. 🎯

🚀 Features
User-Friendly Interface: A simple and intuitive UI built with Streamlit.
Accurate Sentiment Analysis: Utilizes a powerful ensemble model (BaggingClassifier with SVC) for robust and accurate sentiment prediction.
Interactive Predictions: Enter any text and instantly receive the sentiment prediction.
Pre-Trained Model: The model is trained on the twitter_samples dataset from NLTK, providing high-quality results.
📂 Dataset
We used the twitter_samples dataset from the Natural Language Toolkit (NLTK) library, which contains positive and negative tweets for sentiment analysis. This dataset is ideal for training binary sentiment classification models.

🛠️ How to Build a Bag of Words Model
To achieve robust sentiment analysis, we built a Bag of Words (BoW) model following these steps:

Subsetting the Dataset: Selecting only the relevant portions of the dataset (positive and negative tweets).
Plotting Word Frequencies and Removing Stopwords: Visualizing the most common words and removing irrelevant words (stopwords) to improve model performance.
Tokenization: Breaking down text into individual tokens (words) to create a vocabulary.
Stemming: Reducing words to their root form (e.g., "running" to "run") to avoid redundancy.
Lemmatization: Converting words to their base form (e.g., "better" to "good") to improve the model's understanding.
🏗️ Installation and Setup
Follow these steps to set up the project locally:

Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/your-repo.git
cd your-repo
Install Required Packages:

Ensure you have Python installed, then run:

bash
Copy code
pip install -r requirements.txt
Download NLTK Data:

Run the Python script to download the twitter_samples dataset:

python
Copy code
import nltk
nltk.download('twitter_samples')
Run the Streamlit App:

Start the app using the Streamlit command:

bash
Copy code
streamlit run app.py
Enjoy! 🎉 Open your web browser and go to http://localhost:8501 to interact with the sentiment analysis app.

📊 Model Overview
We use a BaggingClassifier with an SVC (Support Vector Classifier) base estimator to perform sentiment analysis. This combination offers high accuracy, robustness against overfitting, and improved generalization.

🔍 Model Details:
Estimator: SVC with a linear kernel
Ensemble Method: Bagging with 10 estimators
Vectorization: TF-IDF to convert text into numerical form
📄 Example Usage
Here's an example of how to use the app:

Input Text: "I had an amazing day at the park today. The weather was perfect, with a clear blue sky and a gentle breeze."
Click 'Predict Sentiment' Button.
Result: Positive sentiment detected! 🌞
📖 How It Works
The sentiment analysis model works in the following steps:

Data Preprocessing: Load and clean the data.
TF-IDF Vectorization: Convert text into numerical features using TF-IDF.
Model Training: Train the BaggingClassifier with SVC on the training data.
Prediction: Use the trained model to predict the sentiment of user-input text.
📈 Future Improvements
Extend the model to handle multi-class sentiment (e.g., neutral, very positive, very negative).
Use a more extensive and varied dataset for better generalization.
Enhance the UI with visualizations like word clouds and sentiment trend graphs.
🌐 Contributing
Feel free to open issues, submit pull requests, and contribute to enhancing this project. Contributions, feedback, and suggestions are always welcome! 🙌

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🔗 References
Scikit-Learn Documentation
Streamlit Documentation
NLTK Documentation
📧 Contact
If you have any questions, feel free to reach out!

Happy Coding! 😊
