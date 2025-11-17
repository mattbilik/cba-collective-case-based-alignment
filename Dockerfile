    # Use a Miniconda base image
    FROM continuumio/miniconda3

    # Set environment variables (optional, but good practice)
    ENV PYTHONDONTWRITEBYTECODE 1
    ENV PYTHONUNBUFFERED 1

    # Set the working directory inside the container
    # WORKDIR /app

    # Copy the environment.yml file
    COPY environment.yml .

    # Create the Conda environment
    RUN conda env create -f environment.yml

    # Activate the Conda environment for subsequent commands
    SHELL ["conda", "run", "-n", "ccai", "/bin/bash", "-c"]

    # Copy your application code
    COPY . .

    # Install any additional pip packages if necessary (after activating conda env)
    RUN pip install -r requirements.txt


    # Train model
    CMD ["python", "hf_rlaif.py"]

