
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Fix for libgomp: Invalid value for environment variable OMP_NUM_THREADS
ENV OMP_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface_hub

COPY progressive_llm_cognitive_hf.py .

# Esegui un server HTTP fittizio in background per superare l'health check di Hugging Face (porta 7860)
# ed esegui lo script di training in foreground
CMD python -m http.server 7860 & python progressive_llm_cognitive_hf.py
