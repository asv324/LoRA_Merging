FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Core ML / QLoRA
RUN pip3 install "transformers==4.46.0"
RUN pip3 install "datasets==3.0.0"
RUN pip3 install "peft==0.13.0"
RUN pip3 install "bitsandbytes==0.44.1"
RUN pip3 install "accelerate==1.0.0"
RUN pip3 install "evaluate==0.4.3"
RUN pip3 install "safetensors==0.4.5"

# Scientific computing (also used for GPA alignment)
RUN pip3 install "numpy==1.26.4"
RUN pip3 install "scipy==1.14.1"
RUN pip3 install "scikit-learn==1.5.2"

# Utilities
RUN pip3 install "wandb==0.18.5"
RUN pip3 install "matplotlib==3.9.2"
RUN pip3 install "seaborn==0.13.2"
RUN pip3 install "pandas==2.2.3"
RUN pip3 install "tqdm==4.66.6"