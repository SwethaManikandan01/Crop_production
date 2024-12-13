from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import io
import base64

app = Flask(__name__)

# Load the trained model and data
model = load_model('crop_production_multi_class_model.h5')
data = pd.read_csv("Crop production dataset.csv").dropna()

# Initialize LabelEncoders and preprocess data for prediction and visualization
label_encoder_crop = LabelEncoder()
data['Crop'] = data['Crop'].str.lower()  # Ensure all crop names are lowercase in the dataset
label_encoder_crop.fit(data['Crop'])
label_encoder_state = LabelEncoder().fit(data['State_Name'])
label_encoder_season = LabelEncoder().fit(data['Season2'])

# Select only numeric columns for scaling
numeric_columns = ['Area', 'Crop_Year']
scaler_X = StandardScaler().fit(data[numeric_columns])

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    crop_name = request.json.get('crop').strip().lower()
    print("Received crop name for prediction:", crop_name)

    # Ensure crop name is in the label encoder's classes
    if crop_name not in label_encoder_crop.classes_:
        print("Error: Crop not found in the dataset.")
        return jsonify({'error': 'Crop not found in the dataset.'}), 400

    # Encode the crop name
    crop_encoded = label_encoder_crop.transform([crop_name])[0]
    print("Encoded crop name for prediction:", crop_encoded)

    # Placeholder values for other features for prediction (customize as needed)
    X = np.array([[crop_encoded, 0, 0, 2021, 0, 0]])  # Adjust encoding if other features are needed
    X = scaler_X.transform(X[:, [0, 3]])  # Only transform numeric columns (Area, Crop_Year)

    prediction = model.predict(X)
    class_prediction = np.argmax(prediction, axis=1)[0]
    print("Prediction result:", class_prediction)

    return jsonify({'prediction': int(class_prediction)})

# Visualization route
@app.route('/visualize', methods=['POST'])
def visualize():
    crop_name = request.json.get('crop').strip().lower()
    print("Received crop name for visualization:", crop_name)

    

    # Filter data for the specified crop
    data_filtered = data[data['Crop'] == crop_name]
    if data_filtered.empty:
        print("Error: No data found for this crop.")
        return jsonify({'error': 'No data found for this crop.'}), 400

    # Aggregate data for highest production state and season
    highest_production_state = data_filtered.groupby('State_Name')['Production Rate'].sum().idxmax()
    highest_production_season = data_filtered.groupby('Season2')['Production Rate'].sum().idxmax()

    # Create knowledge graph
    G = nx.Graph()
    G.add_node(f"Crop: {crop_name}", size=5000)

    for _, row in data_filtered.iterrows():
        state_name = row['State_Name']
        season_name = row['Season2']
        G.add_node(state_name, size=3000)
        G.add_node(season_name, size=3000)
        G.add_edge(f"Crop: {crop_name}", state_name, label="grows in")
        G.add_edge(f"Crop: {crop_name}", season_name, label="grows in")

    G.add_node("High Production Rate", size=3000)
    G.add_edge("High Production Rate", highest_production_state, label="highest in")
    G.add_edge("High Production Rate", highest_production_season, label="most in")

    # Generate and save the knowledge graph
    pos = nx.spring_layout(G, k=0.5, seed=42)
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", width=2)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
    plt.title(f"Knowledge Graph for Crop: {crop_name}", fontsize=15)

    # Save the plot as a PNG image and encode to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode("utf-8")

    print("Knowledge graph generated successfully for:", crop_name)
    return jsonify({'graph': graph_base64})

# Home route
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

  
