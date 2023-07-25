# Jaxformer

JAX library for training of large language models with data and model parallelism based on the pjit() operator on TPU-v3/v4.

## Citation

Please cite:
```bibtex
@misc{Jaxformer,
  title={Jaxformer: A minimal library for training LLMs on TPU},
  author={Nijkamp, Erik},
  howpublished = {\url{https://github.com/salesforce/jaxformer}},
  year={2022}
}
```

Acknowledgments: Ben Wang, James Bradbury, Zak Stone, Bo Pang.

## Models

* CodeGen ([Paper](https://arxiv.org/abs/2203.13474)) ([Code](https://github.com/salesforce/codegen))
* ProGen2 ([Paper](https://arxiv.org/abs/2206.13517)) ([Code](https://github.com/salesforce/progen/tree/main/progen2))

### CodeGen

`350M`

```
gs://sfr-codegen-research/checkpoints/codegen-350M-nl/350000
gs://sfr-codegen-research/checkpoints/codegen-350M-multi/150000
gs://sfr-codegen-research/checkpoints/codegen-350M-mono/150000
```

`2B`

```
gs://sfr-codegen-research/checkpoints/codegen-2B-nl/350000
gs://sfr-codegen-research/checkpoints/codegen-2B-multi/150000
gs://sfr-codegen-research/checkpoints/codegen-2B-mono/100000
```

`6B`

```
gs://sfr-codegen-research/checkpoints/codegen-6B-nl/350000
gs://sfr-codegen-research/checkpoints/codegen-6B-multi/100000
gs://sfr-codegen-research/checkpoints/codegen-6B-mono/140000
```

## Sanity TPU

```sh
import jax
jax.devices()
device_count = jax.device_count()
local_device_count = jax.local_device_count()
xs = jax.numpy.ones(jax.local_device_count())
r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
print('global device count:', jax.device_count())
print('local device count:', jax.local_device_count())
print('pmap result:', r)

gcloud compute tpus tpu-vm ssh erik.nijkamp@sfr-erik.nijkamp-tpu-v3-128-us-east1-d-1 --zone=us-east1-d --internal-ip --worker=all --command="pip install 'jax[tpu]==0.3.16' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
gcloud compute tpus tpu-vm scp test.py erik.nijkamp@sfr-erik.nijkamp-tpu-v3-128-us-east1-d-1:/home/erik.nijkamp/ --zone=us-east1-d --internal-ip --worker=all
gcloud compute tpus tpu-vm ssh erik.nijkamp@sfr-erik.nijkamp-tpu-v3-128-us-east1-d-1 --zone=us-east1-d --internal-ip --worker=all --command="python3 /home/erik.nijkamp/test.py"
```

## Training

### Mode 1: CPU local

```sh
brew install python@3.9
apt install --yes python3.9 python3.9-venv

git clone https://<username>:<secret>@github.com/salesforce/jaxformer.git/
cd jaxformer

python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt

python3 -m jaxformer.train --config config/debug_cpu.json
```

### Mode 2: TPU local

```sh
gcloud compute tpus list --zone=europe-west4-a

gcloud compute tpus tpu-vm delete sfr-erik.nijkamp-tpu-v3-8-europe-west4-d-1 --zone=europe-west4-a --quiet

gcloud compute tpus tpu-vm create sfr-erik.nijkamp-tpu-v3-8-europe-west4-d-1 --zone=europe-west4-a --accelerator-type=v3-8 --version=v2-alpha

gcloud compute tpus tpu-vm ssh sfr-erik.nijkamp-tpu-v3-8-europe-west4-d-1 --zone=europe-west4-a --project <project> --worker 0

export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/legacy_credentials/<username>/adc.json
export GCLOUD_PROJECT=<project>

git clone https://<username>:<secret>@github.com/salesforce/jaxformer.git/
cd jaxformer

./jaxformer/env/env_tpu_v3.sh
pip install -r requirements.txt

source .venv/bin/activate

python3
import jax
jax.devices()
quit()

python3 -m jaxformer.train --config config/debug_tpu_v3_8.json
```

### Mode 3: TPU remote

```sh
gcloud beta compute --project=<project> instances create sfr-<username>-cpu-small-us-east1-d-1 --zone=us-east1-d --machine-type=e2-standard-4 --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=<account> --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --image=ubuntu-minimal-2004-focal-v20210720 --image-project=ubuntu-os-cloud --boot-disk-size=50GB --boot-disk-type=pd-balanced --boot-disk-device-name=sfr-cpu-small --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any

gcloud beta compute ssh sfr-<username>-cpu-small-us-east1-d-1 --project=<project> --zone=us-east1-d

sudo apt update
sudo apt install --yes git screen python3.9 python3.9-venv

screen -S codegen_350M_nl

curl https://sdk.cloud.google.com | bash
source ~/.bashrc
gcloud init
ssh-keygen -t rsa -f ~/.ssh/google_compute_engine -N ''

export WANDB_API_KEY=<secret>
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/legacy_credentials/<username>/adc.json
export GCLOUD_PROJECT=<project>

git clone https://<username>:<secret>@github.com/salesforce/jaxformer.git/
cd jaxformer

python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt

python3 -m jaxformer.train --config config/codegen_350M_nl.json

gcloud compute tpus tpu-vm ssh sfr-erik.nijkamp-tpu-v3-64-us-east1-d-1 --zone us-east1-d --internal-ip --worker=0
```


## Fine-tuning

### TPU fine-tune

```sh
gcloud beta compute --project=<project> instances create sfr-<username>-cpu-small-us-east1-d-1 --zone=us-east1-d --machine-type=e2-standard-4 --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=<account> --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --image=ubuntu-minimal-2004-focal-v20210720 --image-project=ubuntu-os-cloud --boot-disk-size=50GB --boot-disk-type=pd-balanced --boot-disk-device-name=sfr-cpu-small --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any

gcloud beta compute ssh sfr-<username>-cpu-small-us-east1-d-1 --project=<project> --zone=us-east1-d

sudo apt update
sudo apt install --yes git screen python3.9 python3.9-venv

screen -S codegen_350M_mono

curl https://sdk.cloud.google.com | bash
source ~/.bashrc
gcloud init
ssh-keygen -t rsa -f ~/.ssh/google_compute_engine -N ''

export WANDB_API_KEY=<secret>
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/legacy_credentials/<username>/adc.json
export GCLOUD_PROJECT=<project>

git clone https://<username>:<secret>@github.com/salesforce/jaxformer.git/
cd jaxformer

python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt

python3 -m jaxformer.train --config config/codegen_350M_multi.json

gcloud compute tpus tpu-vm ssh sfr-erik.nijkamp-tpu-v3-64-us-east1-d-1 --zone us-east1-d --internal-ip --worker=0
```

### A100 fine-tune

```sh
apt install python3.8 python3.8-venv python3.8-dev

curl https://sdk.cloud.google.com | bash
source ~/.bashrc
gcloud init

export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/legacy_credentials/<username>/adc.json
export GCLOUD_PROJECT=<project>

python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.21.1 datasets==1.16.1 deepspeed==0.7.0 tensorflow-cpu==2.5.0

pip install -e .

deepspeed --num_gpus=1 jaxformer/hf/train.py
```

## Conversion
```
python3 -m jaxformer.hf.convert --config=config/codegen_1B_mono.json --step=150000
```

## Features

v1
- Data
   -  Stateful resumable data loading based on tfrecords without skip()
- TPU
   - Provisioning of TPU clusters and virtual environment
   - Code paths for both TPU-v3 and TPU-v4
   - ...
- Parallelism
   - Push-based single port TCP/IP protocol for orchestration and data-parallelism
   - Megatron pjit() based sharding pattern across TPU boards for up to 6B parameter LLMs
   - xmap() emulation mode through pjit() sharding
   - Distributed checkpointing with full state recovery
   - scan() for time-efficient jit'ing
   - ...
- Debugging
   - Abstraction layer for local/remote workers
   - Local CPU debugging with TPU emulation
   - Mock data iterators
   - ...
- Training
   - Fully resumable state and checkpointing
   - WandB integration
   - ...
