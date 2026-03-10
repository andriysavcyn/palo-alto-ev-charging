EV Charging Station Analysis & Clustering

This project uses Machine Learning (K-Means) to segment electric vehicle charging sessions and identify business-critical patterns, such as "Overstayers" (drivers blocking stations after charging).

Data Sources: Data were taken from Kaggle competition: https://www.kaggle.com/datasets/venkatsairo4899/ev-charging-station-usage-of-california-city?resource=download

Project Structure:
    app/: Contains Python file which has a code with streamlit library which create a website.
    data/: Contains raw data.
    models/: Contains 2 Python Pickle Formats, that have saved model and scale.
    notebooks/: Contains Jupyter Notebook with data analysis and model building.

Features
- **Data Analysis**: Insightful EDA of EV charging sessions.

The first step was to conduct deep Exploratory Data Analysis (EDA). I ​​analyzed over 200,000 charging sessions to find patterns between connection time and actual energy consumed.
Anomaly Detection: Sessions were found where the car occupied the station for more than 24 hours, while consuming minimal energy.
Correlations: The analysis showed that for most users there is a linear relationship, but it is the deviations from it (outliers) that are of greatest interest to the business.

- **Clustering Model**: K-Means algorithm to categorize user behavior into 4 logical groups.

For segmentation, I used the unsupervised learning algorithm K-Means. Elbow Method: Helped determine the optimal number of clusters (K=4), where the intra-cluster error (WCSS) is minimized. Business logic: The model divided users into 4 types: from "fast customers" to "overstayers", which allows station owners to implement flexible tariffs.

- **Web Interface**: Interactive Streamlit app for real-time driver behavior prediction.

I developed an interactive UI that allows any station manager to test the model without writing any code.
Real-time Prediction: The user enters time and energy using sliders, and the model instantly outputs the driver category.
Visual cues: The application uses color-coded statuses (Success, Info, Warning, Error) to quickly interpret the results.

- **Dockerized**: Fully containerized for consistent deployment anywhere.

The project is completely packaged in a Docker container, which solves the "it works on my machine" problem.
Portability: Thanks to the Dockerfile and requirements.txt, the application runs in an identical environment on any server.
Optimization: The lightweight base image python:3.11-slim is used to reduce the size of the container.

Tech Stack
- **Language**: Python 3.11
- **ML**: Scikit-learn, Pandas, NumPy
- **App**: Streamlit
- **DevOps**: Docker

Business Insights

The model identifies 4 key clusters:
Coffee break: Short sessions, low energy.
Standard shopping: Balanced time and energy
Heavy Users: Long sessions with high energy consumption.
Overstayers: Extremely long parking time with minimal energy intake.

Final Results: EV Charging Behavior Analysis
Based on the final K-Means model (K=4), the following segmentation and business insights were obtained:

Cluster Distribution & Statistics:
Cluster 2 (103,345 sessions): "Coffee break" — Short, high-turnover sessions.
Cluster 3 (100,947 sessions): "Standard users" — Typical shopping/errand behavior.
Cluster 0 (45,606 sessions): "Heavy Users" — Deep charging sessions for long-distance drivers.
Cluster 1 (9,127 sessions): The Overstayers 🚩 — Smallest but most critical group.

Key Findings:
Resource Blocking: The model successfully isolated over 9,000 sessions where vehicles were "idling" for an average of 7 hours after the charging process was likely complete.
Feature Importance: Connection Duration proved to be the most significant factor in identifying anomalies, contributing more to the segmentation than the total energy consumed.
Business Impact: Implementing Idle Fees for Cluster 1 could potentially free up enough station capacity to serve an additional 15,000+ "Standard" customers per year.

Proposed Solutions:
Idle Fees: Automated billing after a 30-minute grace period for fully charged cars.
Push Notifications: Real-time alerts via the mobile app when charging stops.
Dynamic Pricing: Higher rates for peak hours to discourage long-term parking.

How to Run with Docker

1. **Build the image**:
   '''bash
   docker build -t ev-charging-app .
2. **Run the container**:
    '''bash
    docker run -p 8501:8501 ev-charging-app
3. **Open http://localhost:8501 in your browser.**