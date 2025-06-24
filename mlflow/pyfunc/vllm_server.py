from mlflow.utils.file_utils import path_to_local_file_uri
import logging
import os
from typing import Optional
import shlex

_logger = logging.getLogger(__name__)


class VllmServer():
    def __init__(self, config):
        self.vllm_ops = config.get("vllm_ops")

    def get_cmd(
        self,
        model_uri: str,
        port: Optional[int] = None,
        host: Optional[str] = None,
        timeout: Optional[int] = None,
        nworkers: Optional[int] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> tuple[str, dict[str, str]]:

        local_uri = path_to_local_file_uri(model_uri)[5:]

        cmd = f"vllm serve {local_uri} "

        args = []
        if host:
            args.append(f"--host {shlex.quote(host)}")

        if port:
            args.append(f"--port {port}")

        if self.vllm_ops:
            args.append(self.vllm_ops)

        cmd += ' '.join(args)

        cmd_env = os.environ.copy()
        return cmd, cmd_env
