import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    df = pd.read_csv(filepath, index_col='customerID') # Load the data frame from filepath
    return df # return the data frame

def make_predictions(df):
    model = load_model('best_churn_model') # Load the best model which is saved as pickle file
    predictions = predict_model(model, data=df) # make predictions using the loaded model with the given data frame
    if 'prediction_label' in predictions.columns: # check if the 'prediction_label' column is present
        predictions.rename(columns={'prediction_label': 'Churn_prediction'}, inplace=True) # rename the column as 'Churn_prediction' to more understanding.
        predictions['Churn_prediction'].replace({1: 'Churn', 0: 'Not Churn'}, inplace=True) # replace the values 1 and 0 with 'Churn' and 'Not Churn'
        return predictions['Churn_prediction'] # return the 'Churn_prediction' column
    else: # if the 'prediction_label' column is not present
        raise KeyError("The 'prediction_label' column was not found in the predictions DataFrame") # raise an error if the 'prediction_label' column is not present

if __name__ == "__main__":
    df = load_data('data/new_churn_data.csv') # call the load_data function by passing the file path
    predictions = make_predictions(df) # call the make_predictions function by passing the data frame
    print(predictions)