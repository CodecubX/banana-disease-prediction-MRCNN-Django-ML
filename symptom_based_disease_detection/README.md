# Banana Disease Prediction Model
This is a machine learning classifier to predict the harvest of a banana plant. The model uses a dataset containing information about banana plants, including their agro-climatic region, fertilizer type, soil pH, amount of sunlight received, watering schedule, and other factors. The goal is to predict the yield of bananas produced by each plant.

##Installation
To use this project, clone the repository to your local machine:

> `git clone <url> `
> 
Then, navigate to the project directory and install the required dependencies:

> `cd banana-disease-prediction`
> 
> `pip install -r requirements.txt`

## Usage
To use the model, open `banana-disease-prediction.ipynb` in Jupyter Notebook. The notebook contains code for training 
the model and making predictions based on input data.

## Dataset
The dataset used to train the model contains information about `Disease/Pest` and `Description` about symptoms.

## Model
This application uses **text similarity** to predict the disease or pest affecting the banana plant based on its symptoms.
It uses `TfidfVectorizer`, `word_tokenize`, `WordNetLemmatizer` and `TF-IDF scores`.

## License
This project solely belongs to *cubX*.