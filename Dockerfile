FROM continuumio/miniconda3

# Create and change to the app directory.
WORKDIR /usr/src/app

# Copy application dependency manifests to the container image
COPY environment.yml ./

RUN conda install --channel defaults conda python=3.7 --yes
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN conda config --set restore_free_channel true

# initialize shell
RUN echo "conda init bash" > ~/.bashrc

# Install dependencies
RUN conda env create -f environment.yml

# Copy local code to the container image.
COPY . ./

# Make run commands use `bash --login`
SHELL ["/bin/bash", "--login", "-c" ]
RUN echo "conda activate perfume_recommendation" > ~/.bashrc
RUN echo "Make sure flask is installed"
RUN python -c "import flask"
RUN python -m spacy download en_core_web_sm
RUN python preset.py

# Run the web service on container startup.
RUN echo "conda init bash && conda activate perfume_recommendation"  > ~/.bashrc

EXPOSE 8080/tcp
CMD conda activate perfume_recommendation && python app.py