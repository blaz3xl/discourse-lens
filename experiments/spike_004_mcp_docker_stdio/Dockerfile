FROM python:3.14-slim
RUN useradd --create-home --uid 10001 spike
WORKDIR /opt/spike
COPY server.py .
USER spike
ENTRYPOINT ["python", "server.py"]
