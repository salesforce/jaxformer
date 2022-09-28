# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import time
from datetime import datetime
import json

from jaxformer.utils import run_return_code, sh, sh_ret, par_map, print_time


def spawn_tpu_workers(tpu_user, tpu_spawn, tpu_create_env, tpu_name, tpu_tags, tpu_image, tpu_zone, tpu_version, tpu_size, tpu_network, tpu_subnetwork, tpu_worker_port, tpu_delete, tpu_reserved, tpu_internal_ips):

    tpu_home = f'/home/{tpu_user}' if not tpu_user == 'root' else '/root'

    tpu_reserved_arg = '--reserved' if tpu_reserved else ''
    tpu_internal_ip_arg = '--internal-ip' if tpu_internal_ips else ''
    tpu_internal_ips_arg = '--internal-ips' if tpu_internal_ips else ''
    tpu_network_arg = f'--network={tpu_network}' if tpu_network else ''
    tpu_subnetwork_arg = f'--subnetwork={tpu_subnetwork}' if tpu_subnetwork else ''
    tpu_tags_arg = f'--tags={tpu_tags}' if tpu_tags else ''

    if tpu_spawn:
        with print_time('Spawning TPU'):
            tpu_exists = (run_return_code(f'gcloud compute tpus tpu-vm describe {tpu_name} --zone {tpu_zone}') == 0)

            if (not tpu_exists) or (tpu_exists and tpu_delete):

                if (tpu_exists and tpu_delete):
                    print(f'deleting existing {tpu_name}')
                    sh(f'gcloud compute tpus tpu-vm delete {tpu_name} --zone={tpu_zone} --quiet', check_return_code=False)

                print(f'spawning new {tpu_name}')
                sh(f'gcloud compute tpus tpu-vm create {tpu_name} --zone={tpu_zone} --accelerator-type=v{tpu_version}-{tpu_size} --version={tpu_image} {tpu_network_arg} {tpu_subnetwork_arg} {tpu_tags_arg} {tpu_internal_ips_arg} {tpu_reserved_arg}')
                time.sleep(5)

    with print_time('Copying library'):
        sh(f'gcloud compute tpus tpu-vm ssh {tpu_user}@{tpu_name} --zone {tpu_zone} --worker=all {tpu_internal_ip_arg} --command="rm -rf {tpu_home}/jaxformer"', check_return_code=False)
        par_map(sh, [f"gcloud compute tpus tpu-vm scp jaxformer {tpu_user}@{tpu_name}:{tpu_home}/ --zone {tpu_zone} --worker={worker} {tpu_internal_ip_arg} --recurse" for worker in range(tpu_size // 8)])

    if tpu_create_env:
        with print_time('Setting up env'):
            sh(f'gcloud compute tpus tpu-vm ssh {tpu_user}@{tpu_name} --zone {tpu_zone} --worker=all {tpu_internal_ip_arg} --command="bash {tpu_home}/jaxformer/env/env_tpu_v{tpu_version}.sh"')

    with print_time('Starting workers'):
        sh(f'gcloud compute tpus tpu-vm ssh {tpu_user}@{tpu_name} --zone {tpu_zone} --worker=all {tpu_internal_ip_arg} --command "killall -9 python3"', check_return_code=False)
        time.sleep(5)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        sh(f'gcloud compute tpus tpu-vm ssh {tpu_user}@{tpu_name} --zone {tpu_zone} --worker=all {tpu_internal_ip_arg} --command "source .venv/bin/activate; python3 -u -m jaxformer.run.remote.worker --port={tpu_worker_port} &> worker.{timestamp}.out &"')

    with print_time('Gathering endpoints'):
        endpoints_json = sh_ret(f"gcloud compute tpus tpu-vm describe {tpu_name} --zone {tpu_zone} --format='get(networkEndpoints)'")
        print(f'endpoints={endpoints_json}')
        endpoints_ips = [json.loads(s.replace("'", '"'))['ipAddress'] for s in endpoints_json.split(';')]
        assert len(endpoints_ips) == tpu_size // 8

        endpoints = [(worker_ip, tpu_worker_port) for worker_ip in endpoints_ips]

        return endpoints