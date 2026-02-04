FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install torch CPU safely
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install rest of deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY properties.xlsx .
COPY model/ model/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
