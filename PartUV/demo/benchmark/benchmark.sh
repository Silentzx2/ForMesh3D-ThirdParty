# See the python script for more details of the parameters.

python benchmark_dataset.py -i datasets/POT -o output_meshes_python/PartObjaverse-Tiny_mesh-benchmark -fo PartObjaverse-Tiny_mesh-benchmark > PartObjaverse-Tiny_mesh-benchmark.log 2>&1
python eval_generate_report.py --data_dir output_meshes_python/PartObjaverse-Tiny_mesh-benchmark/PartObjaverse-Tiny_mesh-benchmark --output_html PartObjaverse-Tiny_mesh-benchmark.html -f

python benchmark_dataset.py -i datasets/common-meshes -o output_meshes_python/common-meshes-benchmark -fo common-meshes-benchmark > common-meshes-benchmark.log 2>&1
python eval_generate_report.py --data_dir output_meshes_python/common-meshes-benchmark/common-meshes-benchmark --output_html common-meshes-benchmark.html -f


python benchmark_dataset.py -i datasets/Trellis -o output_meshes_python/Trellis-benchmark -fo Trellis-benchmark > Trellis-benchmark.log 2>&1
python eval_generate_report.py --data_dir output_meshes_python/Trellis-benchmark/Trellis-benchmark --output_html Trellis-benchmark.html -f

python benchmark_dataset.py -i  datasets/ABC -o output_meshes_python/ABC-benchmark -fo ABC-benchmark > ABC-benchmark.log 2>&1
python eval_generate_report.py --data_dir output_meshes_python/ABC-benchmark/ABC-benchmark --output_html ABC-benchmark.html -f
