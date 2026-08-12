from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    source = Path(os.environ["CHATGPT2API_E2E_SOURCE"]).resolve()
    data_dir = Path(os.environ["CHATGPT2API_E2E_DATA_DIR"]).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))

    import services.config as config_module

    config_module.DATA_DIR = data_dir
    config_module.CONFIG_FILE = data_dir / "config.json"
    config_module.BACKUP_STATE_FILE = data_dir / "backup_state.json"
    config_module.config = config_module.ConfigStore(config_module.CONFIG_FILE)

    import uvicorn
    from api.app import create_app

    uvicorn.run(
        create_app(),
        host="127.0.0.1",
        port=int(os.environ["CHATGPT2API_E2E_PORT"]),
        access_log=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
