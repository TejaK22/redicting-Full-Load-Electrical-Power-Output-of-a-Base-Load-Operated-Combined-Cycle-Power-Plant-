from urllib import request

from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict')
def index():
    return render_template("index.html")
@app.route('/data_predict', methods=['POST'])
def data_predict():
    at = request.form['at']
    v = request.form['v']
    ap = request.form['ap']
    rh = request.form['rh']

    data = [[float(at), float(v), float(ap), float(rh)]]
    model = pickle.load(open('CCPP.pkl', 'rb'))
    prediction = model.predict(data)[0]
    return render_template('index.html' , prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)