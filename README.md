## End-to-End Machine Learning Regression Project
end-to-end Machine Learning regression pipeline featuring automated modular data ingestion, transformation, model training, and hyperparameter tuning across multiple algorithms (XGBoost, CatBoost, Random Forest, etc.) . The optimized production pipeline achieves a peak **R² Score of 0.992**. The application is fully containerised using Docker for seamless local deployment.
## 🚀 Features
- Modular Architecture: Clean separation of ingestion, transformation, and training scripts.
- Multi-Model Tuning: Automated hyperparameter grid search pipeline over 9 different regression algorithms.
- Hardware Agnostic: Automatic fallback configurations between CPU and GPU (cuda:0).
- Robust Logging: Centralised custom exception handling and logging modules for simple debugging.
- Dockerised Deployment: Fully containerised setup for deterministic execution on any machine.

### Option 1: Try the Live Web App (Recommended for Recruiters)
You can test the model's predictions directly in your browser without installing anything.
1. Visit the live Streamlit application: [Insert Link Here]
2. Adjust the input parameters using the sidebar sliders/dropdowns.
3. Click the "Predict" button to see the model's output in real-time.

### Option 2: 💻 Local Setup (Using Python)
Follow these steps to configure and run the project directly on your physical hardware.
### 1. Clone the Repositorybashgit clone
   ```bash
   git clone [https://github.com/AbhishekJadhav015/SmartphonePricePredictionML.git
   ](https://github.com/AbhishekJadhav015/SmartphonePricePredictionML.git
   )
   cd ML_project
   ```
### 2. Create and Activate a Virtual Environment
  ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```
### 3. Install Required Dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
### 4. Execute the Training Pipeline
Run the main pipeline driver file to process data, execute the hyperparameter grid search, and export your production-ready model artifact:
```bash
python src/pipelines/training_pipeline.py
```
The optimized model will automatically save to the "artifacts/model.pkl" directory upon pipeline completion

## 🐳 Running with Docker

You can pull and execute the pre-built, production-ready container image directly from Docker Hub.

https://hub.docker.com/r/abhishekjadhav0015/ml-pipeline-app/tags

### 1. Pull the Image from Docker Hub
```bash
docker pull abhishekjadhav0015/ml-pipeline-app:latest
```

### 2. Run the Containerized Application
Execute the container and map the web port to view the interface locally:
```bash
docker run --rm -p 8501:8501 abhishekjadhav0015/ml-pipeline-app:latest
```
*Once running, navigate to `http://localhost:8501` in your browser.*