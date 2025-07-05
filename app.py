import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# import Mysql.connector

app = Flask(__name__) #Initialize the flask App

# sr = Mysql.connector.connect(username = "root",
#                              password = "root",
#                              host = "localhost")

heart = pickle.load(open('heartd.pkl','rb'))
@app.route('/')

@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/chart')
def chart():
	return render_template('chart.html')

#@app.route('/future')
#def future():
#	return render_template('future.html')    

@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	


#@app.route('/home')
#def home():
 #   return render_template('home.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    return render_template('prediction.html')


#@app.route('/upload')
#def upload_file():
#   return render_template('BatchPredict.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get all form values and convert to float
        form_values = [float(x) for x in request.form.values()]
        
        # 2. Load model and training column structure
        import json
        model = pickle.load(open('heartd.pkl', 'rb'))
        with open('columns.json', 'r') as f:
            model_columns = json.load(f)

        # 3. Convert to DataFrame and reindex to match training
        df = pd.DataFrame([form_values], columns=model_columns)
        df = df.reindex(columns=model_columns, fill_value=0)

        # 4. Predict
        result = model.predict(df)[0]
        output = "Positive" if result == 1 else "Negative"

        # 5. Return prediction
        return render_template('prediction.html', prediction_text=output)

    except Exception as e:
        print("❌ Prediction error:", e)
        return f"Internal Server Error: {e}", 500

    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
