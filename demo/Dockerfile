FROM python:3.12-slim
WORKDIR /app
COPY demo/requirements.txt /app/demo/requirements.txt
RUN pip install --no-cache-dir -r demo/requirements.txt
RUN useradd --uid 1000 --create-home mosaic
COPY --chown=mosaic:mosaic src /app/src
COPY --chown=mosaic:mosaic demo/server.py /app/demo/server.py
COPY --chown=mosaic:mosaic docs /app/docs
USER mosaic
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EXPOSE 7860
CMD ["python", "demo/server.py", "--host", "0.0.0.0", "--port", "7860", "--allowed-origin", "https://gengyuanmax.github.io"]
