"""Start the unchanged upstream TiPToP server with a full, explicit config.

Run in TiPToP's pixi environment:
    pixi run python /path/to/project/scripts/planner_with_config.py \
        --config /path/to/project/configs/tiptop_panda_sim.yml \
        --host 127.0.0.1 --port 8765

All arguments after --config's value, except --check-config, are forwarded to
the upstream server CLI. This selects an experimental condition; it does not
change the default configuration used by the original baseline server.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--check-config', action='store_true')
    args, server_args = parser.parse_known_args()
    from tiptop.config import set_tiptop_cfg_from_file

    cfg = set_tiptop_cfg_from_file(args.config.resolve())
    if args.check_config:
        print(f'Loaded full configuration: robot.type={cfg.robot.type}')
        return
    from tiptop.tiptop_websocket_server import entrypoint

    sys.argv = [sys.argv[0], *server_args]
    entrypoint()


if __name__ == '__main__':
    main()
