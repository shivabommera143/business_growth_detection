Business Growth Detection (Startup Profit Predictor) 
This web app predicts the profit of a startup based on its R&D Spend, Administration Spend, and Marketing Spend.
It uses a Linear Regression model trained on the 50_Startups dataset (Kaggle) to forecast business growth.

The model achieves around 90% accuracy in predicting profits. You can try the live version here: https://business-growth-detection.onrender.com/

How It Works?
User enters values for:
-> R&D Spend 
-> Administration Spend 
-> Marketing Spend

Flask backend passes inputs to the trained regression model. T
he model predicts the expected profit, which is displayed on the web interface.
Model Details 
Algorithm: Linear Regression 
Dataset: 50_Startups.csv 
Accuracy: ~90% (R² Score) 
Hosted using Render
