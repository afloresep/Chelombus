FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy specifically the backend app in charge of the searching (find-cluster)
COPY ./app /app/app

COPY ./data /app/data
 
CMD ["uvicorn", "app.find_cluster:app", "--host", "0.0.0.0", "--port", "8000"]
