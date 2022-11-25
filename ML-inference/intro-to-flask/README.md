# Simple REST API for models
Once a model is trained you want to expose it in some way to make predictions. In some cases you want to use your model programatically in a python or a C++ program, in other cases you want to expose it as a web API (e.g. REST) so that other services can make use of your model.

This repo shows how a simple python webserver such as `Flask` can be used for this (with a faked model which only multiplies the input by 2). A simple library like this can be useful for development or for toy examples, but for real use cases there are model servers that are specifically built for serving machine learning models. These can integrate advanced functionality such as model hot swapping (changing to a new model without any downtime) and model monitoring to predict drift.

Some more advanced model servers to look at are TensorFlow Serving and PyTorch Serving.

## Run the application
First install flask:
```bash
pip install flask
```

Then start the application:
```bash
python app.py
```

There is now a webserver running which you can access by going to the URL `http://localhost:5000` or `http://127.0.0.1:5000` in your browser. The website has a static html page at the root URL and an API at the `/predict` URL. Making a `GET` request to the predict endpoint like `http://127.0.0.1:5000/predict?data=10` will return a json response with the content:
```json
{
  "response": 20
}
```
