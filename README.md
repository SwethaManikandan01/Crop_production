# 🌾 **FNN Crop production rate prediction for farmers using Knowledge Graph**

# 📜 **Project Overview**
This project implements a deep learning-based approach to classify crop production levels (low, medium, high) and visualize insights using a knowledge graph. A user-friendly Flask web interface is designed for user interaction, allowing real-time visualization of crop-specific knowledge graphs based on the input crop name.

# Features
Multi-class classification of crop production using a feed-forward neural network.
Knowledge graph visualization for relationships between crops, seasons, and states.
Flask-based user interface for seamless interaction.
Insights on top-performing states and seasons for a given crop.
Scalable design to incorporate additional data and real-time predictions.

# 📊 **Dataset**
Source: Indian Agriculture Crop Production database.

Key Features:

`Crop Type`, `Season`, `State`, `District`, `Area`, `Crop Year`, `Production Rate`.

Target Variable: Production Class (`Low, Medium, High`).

# ⚙️**System Workflow**
**1)  Data Preprocessing:**
        Handle missing values and normalize categorical data using label encoding.
        Define production classes using quantiles for multi-class classification.
        Scale feature data and one-hot encode the target variable.
        
   ![Crop production Dataset](https://github.com/user-attachments/assets/4cb61081-657e-4383-b28c-5f139b8bdc22)

**2)  Model Development:**
        Train a deep learning model using TensorFlow/Keras with multiple hidden layers and ReLU activation.
        Use early stopping to optimize training and prevent overfitting.
       
   ![Model Training & Evaluation](https://github.com/user-attachments/assets/c306d601-11f5-4bb8-86ff-b9614c4b3c6a)

   # Classification Metrics for each class


   ![Classification Metrics for each class](https://github.com/user-attachments/assets/8d4007ec-4b34-4f6c-b17e-42f3a2847509)
   # Predicted and Actual classes
   ![image](https://github.com/user-attachments/assets/8f10d99b-e4ea-408e-800c-64aaff576981)

   ![Comparison of Predicted and Actual classes](https://github.com/user-attachments/assets/e13cd3c6-588f-4d9d-88fd-642c236540a9)

   # Training and Validation Accuracy over epochs

   ![image](https://github.com/user-attachments/assets/05d1fb7e-ab70-4bd7-a6f6-7245db0f6ef0)

   ![Training and Validation Accuracy over epochs](https://github.com/user-attachments/assets/a987e465-9a59-441c-95be-64fc08b29eb1)

   # Training and Validation loss over epochs

   ![image](https://github.com/user-attachments/assets/7d678a0b-6baf-48f1-b3fc-338ea6988f33)

   
   ![Training and Validation loss over epochs](https://github.com/user-attachments/assets/79a50543-278b-43a4-84d3-72aaefde58b5)

**3)  Knowledge Base Construction:**
        A KB for predicting crop production rates and prices can consist of the following components:

        Entities: Key objects such as crops, years, regions, production rates, and prices.

        Attributes: Characteristics of each entity (e.g., production rate, price, soil type).

        Relationships: Links between entities (e.g., "Region X affects Crop Y's production").

        Rules: Logical inferences or trends derived from historical data and predictive models.
   
   ![Knowledge Base in Tabular Form](https://github.com/user-attachments/assets/4bbaba6f-b9be-48f3-a97b-ac763a33e55b)

        
**4) Knowledge Graph Generation:**
        Use NetworkX to create graphs visualizing crop relationships with states and seasons.
        Highlight the state and season with the highest production for the given crop.
        
   ![Knowledge Graph generation](https://github.com/user-attachments/assets/8f9baa6b-2096-40cd-8c42-bcf68a0c403f)

**5) User Interaction:**
        Develop a Flask-based web application for user input and graph visualization.
   
   ![User Interface](https://github.com/user-attachments/assets/3883a20e-1af8-4745-8fca-7a0fc03d77e5)




# 📚 **Libraries Used**

- **Deep Learning**: TensorFlow, Keras

- **Data Processing**: Pandas, NumPy, Scikit-learn

- **Graph Visualization**: NetworkX, Matplotlib

- **Web Development**: Flask

# 🖥️**System Requirements**
# Hardware:
- **Processor**: Intel Core i5 or higher.
- **RAM**: Minimum 8GB.
- **Storage**: 256GB SSD (minimum).

# Software:
Python 3.x
Libraries: TensorFlow, Pandas, NetworkX, Flask, Matplotlib
Usage Instructions

# Setup:

Clone the project repository.
Install dependencies using pip install -r requirements.txt.
Run the Application:

Start the Flask server with python app.py.
Open the web interface at http://127.0.0.1:5000 or http://192.168.1.120:5000

# Input:
Enter the name of the crop (e.g., "Rice").

![4](https://github.com/user-attachments/assets/27580edc-d9f8-433f-8465-94d47c79f669)


# Output:
View the knowledge graph and insights for the entered crop.

![5](https://github.com/user-attachments/assets/a8adcc91-18a9-49d8-810e-c53752aa5ac5)


# 🔮 **Future Enhancements**
- **Real-Time Data Integration**: Incorporate real-time weather and market data for dynamic predictions.

- **Mobile-Friendly Interface**: Adapt the Flask app for mobile devices to improve accessibility.

- **Expanded Dataset**: Integrate additional crops, soil data, and economic parameters for broader applicability.
