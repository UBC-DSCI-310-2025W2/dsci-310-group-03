FROM quay.io/jupyter/scipy-notebook:python-3.11

COPY environment.yml .

RUN conda env update --name base --file environment.yml && \
    conda clean --all -y