import os

import pytest

from mlflow.pyfunc.vllm_server import VllmServer


@pytest.mark.parametrize(
    "params",
    [
        {"port": 5000, "host": "0.0.0.0", "nworkers": 4},
        {"host": "0.0.0.0", "nworkers": 4},
        {"port": 5000, "nworkers": 4},
        {"port": 5000},
        {"model_name": "mymodel", "model_version": "12"},
        {},
    ],
)
def test_get_cmd(params: dict):
    model_uri = "/foo/bar"
    vllm_ops = "--max-tokens=1024 --gpu-memory-utilization=0.9"

    cmd, cmd_env = VllmServer({
        "vllm_ops": vllm_ops
    }).get_cmd(model_uri=model_uri, **params)

    expected_cmd = f"vllm serve {model_uri}"
    
    if params.get('host'):
        expected_cmd += f" --host={params.get('host')}"
        
    if params.get('port'):
        expected_cmd += f" --port={params.get('port')}"
        
    expected_cmd += f" {vllm_ops}"
    
    assert cmd == expected_cmd

    assert cmd_env == os.environ.copy()
