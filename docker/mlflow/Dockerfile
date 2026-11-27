FROM python:3.12-slim

ARG MLFLOW_VERSION=3.18.0
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}"

# Run as a non-root user; /mlflow holds the SQLite backend store and model artifacts
RUN useradd --create-home --uid 1000 mlflow \
    && mkdir -p /mlflow/artifacts \
    && chown -R mlflow:mlflow /mlflow
USER mlflow
WORKDIR /mlflow

EXPOSE 5000

# --allowed-hosts: other containers reach the server as http://mlflow:5000, so that host
# name must be accepted in addition to localhost (DNS-rebinding protection in MLflow 3).
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--workers", "2", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--allowed-hosts", "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow,mlflow:5000"]
