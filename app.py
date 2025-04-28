from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Pengelompokan Negara Berdasarkan Pola Emisi Karbon dan Konsumsi Energi Global (2001–2021)", 
            description="API digunakan untuk mengelompokkan negara berdasarkan pola emisi karbon dan konsumsi energi global (2001–2021)")

# Load Scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load PCA Model
with open('pca_model.pkl', 'rb') as f:
    pca_model = pickle.load(f)

# Load Clustering Model
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Mapping Penjelasan Cluster
cluster_description = {
    0: "🔵 Cluster 0: Negara dengan emisi karbon rendah dan konsumsi energi rendah.",
    1: "🟢 Cluster 1: Negara dengan emisi karbon sedang dan konsumsi energi moderat.",
    2: "🔴 Cluster 2: Negara dengan emisi karbon tinggi dan konsumsi energi tinggi."
}

# Skema input dari user
class EmissionData(BaseModel):
    Country: str = Field(..., example="Indonesia", description="Nama negara yang akan diprediksi")
    Total: float = Field(..., example=5.12345, description="Total emisi karbon (1.000 (ton CO2))")
    Coal_Ratio: float = Field(..., example=0.6, description="Rasio emisi dari batu bara (0-1)")
    Oil_Ratio: float = Field(..., example=0.2, description="Rasio emisi dari minyak bumi (0-1)")
    Gas_Ratio: float = Field(..., example=0.1, description="Rasio emisi dari gas alam (0-1)")
    Cement_Ratio: float = Field(..., example=0.1, description="Rasio emisi dari industri semen (0-1)")

# Fungsi Preprocessing
def preprocess_input(data: EmissionData):
    df = pd.DataFrame([{
        "Total": data.Total,
        "Coal_Ratio": data.Coal_Ratio,
        "Oil_Ratio": data.Oil_Ratio,
        "Gas_Ratio": data.Gas_Ratio,
        "Cement_Ratio": data.Cement_Ratio
    }])

    # Scaling
    scaled = scaler.transform(df)

    # PCA transform
    reduced = pca_model.transform(scaled)

    return reduced

@app.get("/")
def read_root():
    return {"message": "🚀 Carbon Emission Clustering API is Running! Visit /docs untuk testing."}

@app.post("/predict", summary="Prediksi Cluster Negara")
def predict_cluster(data: EmissionData):
    processed = preprocess_input(data)
    cluster = kmeans_model.predict(processed)[0]

    return {
        "Negara": data.Country,
        "Prediksi Cluster": int(cluster),
        "Deskripsi Cluster": cluster_description.get(int(cluster), "Cluster tidak diketahui"),
        "Interpretasi": "Cluster ini menunjukkan kategori negara berdasarkan pola emisi karbon dan konsumsi energi."
    }