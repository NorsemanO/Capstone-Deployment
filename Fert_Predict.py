from flask import Flask, request, jsonify
import numpy as np

from joblib import load

# Load the trained model (TensorFlow example)
with open('model_fert_clf.pkl', 'rb') as model_file:
  model = load(model_file)

app = Flask(__name__)
pred_label = {0:'Abnormal', 1:'Normal'}

@app.route('/predict', methods=['POST'])
def predict():
  try:
    # Extract 9 inputs from the request
    input_data = request.get_json(force=True)
    inputs = np.array(input_data['inputs']).reshape(1, -1)

    # Perform prediction
    prediction = model.predict(inputs)

    # Return the prediction

    return jsonify({'prediction': pred_label[prediction.tolist()[0]]})
  except Exception as e:
    return jsonify({'error': str(e)})

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=80)
