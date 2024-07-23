from flask import Flask, request, jsonify
import pickle

# Save the model
model.save('drug_discovery_model.h5')

# Load the model
model = tf.keras.models.load_model('drug_discovery_model.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    compound_smiles = data['compound_smiles']
    target_sequence = data['target_sequence']

    # Compute descriptors
    compound_desc = pd.DataFrame([compute_molecular_descriptors(compound_smiles)])
    protein_desc = pd.DataFrame([compute_protein_descriptors(target_sequence)])
    features = pd.concat([compound_desc, protein_desc], axis=1)

    # Make prediction
    prediction = model.predict(features)
    return jsonify({'interaction_probability': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
