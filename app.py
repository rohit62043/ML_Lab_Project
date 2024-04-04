from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    arr = []
    for i in range(1, 28):
        key = 'a' + str(i)
        data = request.form.get(key)
        arr.append(data)
    arr = np.array([arr])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

# if __name__ == "__main__":
#     app.run(debug=True)
