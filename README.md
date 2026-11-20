# Banking Campaign Response Prediction and Contact Prioritization

## Project Overview
This project uses Machine Learning to predict whether a customer will subscribe to a bank's term deposit after a marketing campaign.

It also ranks customers based on their predicted subscription probability, helping banks prioritize customers when calling resources are limited.

## Objectives
- Predict customer subscription responses (Yes/No).
- Train and compare multiple Machine Learning algorithms.
- Select the best-performing model.
- Estimate subscription probabilities.
- Rank customers based on their likelihood of subscribing.
- Deploy the trained model using FastAPI.
- Expose the API through ngrok for public testing.

## Dataset
**Dataset:** Bank Marketing Dataset

**Source:** UCI Machine Learning Repository

The dataset contains customer information, previous campaign details, and economic indicators.

## Machine Learning
**Selected Model:** Calibrated HistGradientBoosting

The project compares 14 Machine Learning models and selects the final model based on validation performance.

### Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 87.74% |
| Precision | 46.64% |
| Recall | 60.92% |
| F1-score | 52.83% |
| ROC-AUC | 81.65% |
| PR-AUC | 48.80% |

**Classification Threshold:** 0.60

## Contact Prioritization
Customers are ranked by their predicted subscription probability.

This helps banks prioritize customers when they have limited calling capacity.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- FastAPI
- Uvicorn
- Postman
- ngrok
- GitHub
- Kaggle

## Project Structure
```text
banking-campaign-prediction/
├── api.py
├── requirements.txt
├── README.md
├── banking_campaign_model.pkl
└── banking_campaign_prediction.ipynb
```

## API Deployment
The trained model is deployed using FastAPI.

**API Documentation:**  
https://excretion-handball-cartwheel.ngrok-free.dev/docs

The API provides:
- Single customer prediction
- Batch customer prediction
- Subscription probability
- Prediction result (Yes/No)
- Customer priority ranking

Note: The public API is available only while the local FastAPI server and ngrok tunnel are running. The public URL may change when the tunnel restarts.

## Testing
The API was tested using:
- FastAPI Swagger UI
- Postman
- Public ngrok URL

Successful HTTP 200 responses were obtained for single and batch predictions.

## Kaggle Links
**Kaggle Notebook:**  
https://www.kaggle.com/code/vaishnavigadiga/banking-campaign-prediction

**Kaggle Model:**  
https://www.kaggle.com/models/vaishnavigadiga/banking-campaign-prediction/scikitLearn/default

## Limitations
- Predictions depend on the quality and distribution of the training data.
- Model performance may vary for new customer populations.
- Predictions indicate probabilities, not guaranteed customer behavior.
- The API requires the local server and ngrok tunnel to remain active.

## Author
Vaishnavi Gadiga

B.Tech – Computer Science and Engineering (AI & ML)
