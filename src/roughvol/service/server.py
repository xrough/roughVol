"""
RoughPricingService gRPC server entry point.

Usage:
    python -m roughvol.service.server              # default port 50051
    python -m roughvol.service.server --port 50052
    rough-pricing-server                           # via installed script
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Make the generated stubs importable
_GEN_DIR = Path(__file__).resolve().parents[3] / "generated" / "python"
if str(_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GEN_DIR))

import grpc
import rough_pricing_pb2_grpc as pb2_grpc

from roughvol.service.servicer import RoughVolServicer


async def serve(port: int) -> None:
    server = grpc.aio.server()
    pb2_grpc.add_RoughPricingServiceServicer_to_server(RoughVolServicer(), server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logging.info(f"RoughPricingService listening on {listen_addr}")
    await server.start()
    await server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="RoughPricingService gRPC server")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    asyncio.run(serve(args.port))


if __name__ == "__main__":
    main()
