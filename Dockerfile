#FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.2.0-gpu-py310-cu118-ubuntu20.04-ec2
#FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.4.0-gpu-py311-cu124-ubuntu22.04-ec2
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-ec2

RUN pip install torchvision torchaudio torch_geometric torch_scatter --no-deps torch && \
    pip install torch_optimizer pytorch-ranger biopython numpy seaborn matplotlib scikit-learn && \
#    pip install reinmax && \
    pip install jupyterlab && \
    pip install fair-esm --no-deps torch && \
    pip install git+https://github.com/microsoft/protein-sequence-models.git --no-deps torch && \
    pip install git+https://github.com/microsoft/evodiff.git --no-deps torch && \
#    pip install "mamba-ssm==2.1.0" && \
#    pip install tensorboard && \
#    pip install git+https://github.com/Bitbol-Lab/ProtMamba-ssm --no-deps mamba-ssm && \
#    pip install esm && \
    pip cache purge

RUN mkdir -p /root/.cache/torch/hub/checkpoints
COPY msa-oaar-maxsub.tar /root/.cache/torch/hub/checkpoints/
COPY esm_msa1b_t12_100M_UR50S.pt /root/.cache/torch/hub/checkpoints/
#COPY esm_msa1b_t12_100M_UR50S-contact-regression.pt /root/.cache/torch/hub/checkpoints/
#COPY ProtMamba-Long-foundation/ /root/ProtMamba-Long-foundation/
COPY proteinmpnn_weights /root/proteinmpnn_weights

RUN jupyter-lab --generate-config
COPY jupyter_lab_config.py /root/.jupyter/

COPY wrappers /root/wrappers
COPY *.py /root/
COPY *.pth /root/

RUN wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb && \
    apt-get update && \
    apt-get install -y ./mount-s3.deb

RUN echo "user_allow_other" >> /etc/fuse.conf

EXPOSE 8888

WORKDIR /root
USER root

ENTRYPOINT ["jupyter-lab", "--allow-root"]
