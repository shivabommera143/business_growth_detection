from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from mangum import Mangum
import os


os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join("50_Startups.csv")

app = Flask(__name__, template_folder="../templates")

df = pd.read_csv(CSV_PATH)
df = df.drop('State', axis=1)

X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
r2_percentage = int(r2 * 100)
print(f"Model trained. Accuracy: {r2_percentage}%")

# ✅ Routes
@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_profit = None
    if request.method == 'POST':
        R_and_D = float(request.form['R_and_D'])
        Administration = float(request.form['Administration'])
        Marketing = float(request.form['Marketing'])
        new_data = pd.DataFrame(
            [[R_and_D, Administration, Marketing]],
            columns=['R&D Spend', 'Administration', 'Marketing Spend']
        )
        predicted_profit = round(model.predict(new_data)[0], 2)
    return render_template('index.html', prediction=predicted_profit)

@app.route('/favicon.ico')
def favicon():
    return '', 204

handler = Mangum(app)
