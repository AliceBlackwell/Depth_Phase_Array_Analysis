conda install -c conda-forge libpq blas lapack #patchelf
make clean
make serial_nodb

CURRENT_DIR=$(pwd)
STATIC_PATH=$(realpath "$CURRENT_DIR/../etc")
echo $STATIC_PATH
echo "export QETC=\"$STATIC_PATH\"" >> ~/.bashrc
echo export LD_LIBRARY_PATH=\"$(dirname $(dirname $(which python)))/lib:\$LD_LIBRARY_PATH\" >> ~/.bashrc
source ~/.bashrc
mv iscloc_nodb ../..
