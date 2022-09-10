cargo run --release --bin offline_render datasets/neuschwanstein/neuschwanstein_8192max_sh.bin figures/threshold_renders/neuschwanstein_2.png  -w 720 -h 540  --lod-threshold 2 -x="-2.899" -y="30.564" -z="43.745"

cargo run --release --bin offline_render datasets/neuschwanstein/neuschwanstein_8192max_sh.bin figures/threshold_renders/neuschwanstein_8.png  -w 720 -h 540  --lod-threshold 8 -x="-2.899" -y="30.564" -z="43.745"

cargo run --release --bin offline_render datasets/neuschwanstein/neuschwanstein_8192max_sh.bin figures/threshold_renders/neuschwanstein_16.png  -w 720 -h 540  --lod-threshold 16 -x="-2.899" -y="30.564" -z="43.745"


cargo run --release --bin offline_render datasets/neuschwanstein/neuschwanstein_8192max_sh.bin figures/threshold_renders/neuschwanstein_64.png  -w 720 -h 540  --lod-threshold 64 -x="-2.899" -y="30.564" -z="43.745"
