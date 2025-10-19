from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from mangum import Mangum
import os

app = Flask(__name__, template_folder="../templates")

# Load dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../50_Startups.csv"))
df = df.drop('State', axis=1)

# Train model
X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy (printed only during build)
y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
r2_percentage = int(r2 * 100)
print(f"Model trained. Accuracy: {r2_percentage}%")

@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_profit = None
    if request.method == 'POST':
        R_and_D = float(request.form['R_and_D'])
        Administration = float(request.form['Administration'])
        Marketing = float(request.form['Marketing'])
        new_data = pd.DataFrame([[R_and_D, Administration, Marketing]],
                                columns=['R&D Spend', 'Administration', 'Marketing Spend'])
        predicted_profit = round(model.predict(new_data)[0], 2)
    return render_template('index.html', prediction=predicted_profit)

# Required handler for Vercel
handler = Mangum(app)
