#cargo run --release --bin offline_render datasets/neuschwanstein/neuschwanstein_8192max.bin figures/dataset_renders/neuschwanstein.png  -w 1280 -h 960  --lod-threshold 0 -x="-2.899" -y="30.564" -z="43.745"
cargo run --release --bin offline_render datasets/navvis_tum_audimax/navvis_tum_audimax_8192max_sh.bin figures/dataset_renders/audimax.png -w 1280 -h 960  --lod-threshold 0 -x="29.799" -y="-4.283" -z="-3.400"
#cargo run --release --bin offline_render datasets/navvis_office/navvis_office_8192max_sh.bin figures/dataset_renders/office.png -w 1280 -h 960  --lod-threshold 0 -x="7.754" -y="-1.470" -z="-34.162"
#cargo run --release --bin offline_render datasets/kaiserburg/kaiserburg_8192max_sh.bin figures/dataset_renders/kaiserburg.png -w 1280 -h 960  --lod-threshold 0 -x="-2.899" -y="30.564" -z="43.745"