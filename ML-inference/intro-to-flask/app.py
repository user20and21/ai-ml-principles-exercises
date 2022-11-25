from flask import Flask, render_template, request, jsonify

class Model:
    """A very simple prediction model
    """
    def __init__(self, multiplier):
        self._m = multiplier

    def __call__(self, value):
        return self._m * value

app = Flask(__name__)
model = None

@app.get("/")
def index_get():
    return render_template("home.html")

@app.get("/predict")
def predict():
    data = request.args.get("data")

    if data is not None:
        try:
            value = int(data)
            message = {"response": 2*value}
        except ValueError:
            message = {"error": "data is not an integer"}
    else:
        message = {"error": "missing parameter 'data'"}

    return jsonify(message)


if __name__ == "__main__":
    model = Model(2)
    app.run(debug=True)
