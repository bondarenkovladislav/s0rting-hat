from flask import Flask
from flask import request, jsonify
from classifier.compose import predictLabels
import numpy as np
import json

app = Flask(__name__)
import sys
#
@app.route('/predict', methods = ['POST'])
def predict():
    try:
        json_string = request.get_data(as_text=True)
        obj = json.loads(json_string)
        arr = np.array(obj)
        return predictLabels(arr)
    except:
        print("Unexpected error:", sys.exc_info())

print('body')
if __name__ == '__main__':
    app.run()
