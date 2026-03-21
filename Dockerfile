FROM quay.io/jupyter/scipy-notebook:python-3.11

COPY environment.yml .

RUN conda env update --name base --file environment.yml && \
    conda clean --all -y

# Install Quarto for report rendering
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl gdebi-core && \
    curl -LO https://github.com/quarto-dev/quarto-cli/releases/download/v1.6.43/quarto-1.6.43-linux-amd64.deb && \
    gdebi --non-interactive quarto-1.6.43-linux-amd64.deb && \
    rm quarto-1.6.43-linux-amd64.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
USER ${NB_UID}