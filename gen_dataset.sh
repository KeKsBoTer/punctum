FOLDER=
MAX_OCTANT_SIZE=8192
SAMPLE_RATE=1
if [ "$SAMPLE_RATE" == "1" ];
then
   SEP=""
else
   SEP=${SAMPLE_RATE}_
fi

SUFFIX="${SEP}${MAX_OCTANT_SIZE}max"

cargo run --release --bin \
   build_octree datasets/neuschwanstein/3DRM_neuschwanstein_original.las \
   datasets/neuschwanstein/neuschwanstein_$SUFFIX.bin \
   --max-octant-size $MAX_OCTANT_SIZE \
   --flip-yz \
   --sample-rate $SAMPLE_RATE

if [ $? -eq 0 ]
then
cargo run --release --bin gen_dataset datasets/neuschwanstein/neuschwanstein_${SUFFIX}.bin datasets/neuschwanstein/octants_${SUFFIX}                  
fi

if [ $? -eq 0 ]
then
python3 pointnet/calc_sh.py  datasets/neuschwanstein/octants_${SUFFIX} datasets/neuschwanstein/octants_${SUFFIX}_sh --l_max 10
fi

if [ $? -eq 0 ]; then
   echo OK
else
   python3 pointnet/calc_sh.py  datasets/neuschwanstein/octants_${SUFFIX} datasets/neuschwanstein/octants_${SUFFIX}_sh --l_max 10
fi


if [ $? -eq 0 ]
then
cargo run --release --bin merge_sh datasets/neuschwanstein/neuschwanstein_${SUFFIX}.bin datasets/neuschwanstein/octants_${SUFFIX}_sh datasets/neuschwanstein/neuschwanstein_${SUFFIX}_sh.bin
fi