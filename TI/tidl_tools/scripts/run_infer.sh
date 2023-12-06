CURDIR=`pwd`

export SOC=am68pa
export TIDL_TOOLS_PATH=$(pwd)/tidl_tools
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH
export ARM64_GCC_PATH=$(pwd)/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu

    cd $CURDIR/examples/osrt_python/ort
    #python3 onnxrt_ep.py -c
    python3 onnxrt_seed.py
    #python3 onnxrt_ep.py -d