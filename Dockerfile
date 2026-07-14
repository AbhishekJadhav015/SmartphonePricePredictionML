# 1. Use the official Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install critical OS-level developer tools needed by CatBoost/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. FIX: Copy ALL project files upfront so '-e .' can find setup configurations
COPY . .

# 5. Upgrade package managers and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 6. Expose the deployment network port
EXPOSE 8501

# 7. Command to execute the Streamlit script
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
