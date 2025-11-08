# !/bin/bash
export LD_LIBRARY_PATH="./hexl-bindings/hexl/build/hexl/lib:$(pwd):/usr/local/lib"

RUSTFLAGS=-Awarnings cargo "$@"
