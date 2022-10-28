#cargo run --release --bin build_octree datasets/kaiserburg/3DRM_kaiserburg_d002.las datasets/kaiserburg/kaiserburg_8192max_mean.bin  --max-octant-size 8192 --flip-yz  --no-pb
cargo run --release --bin build_octree datasets/navvis_tum_audimax/navvis_tum_audimax.ply datasets/navvis_tum_audimax/navvis_tum_audimax_8192max_mean.bin --max-octant-size 8192 --flip-yz  --no-pb
cargo run --release --bin build_octree datasets/navvis_office/navvis_office.ply datasets/navvis_office/navvis_office_8192max_mean.bin --max-octant-size 8192 --flip-yz  --no-pb
#cargo run --release --bin build_octree datasets/neuschwanstein/3DRM_neuschwanstein_original.las datasets/neuschwanstein/neuschwanstein_8192max_mean.bin --max-octant-size 8192 --flip-yz  --no-pb
