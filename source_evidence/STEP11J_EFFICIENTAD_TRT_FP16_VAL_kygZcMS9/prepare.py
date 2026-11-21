import sys
import textwrap
from pathlib import Path

source, target = map(Path, sys.argv[1:])
code = source.read_text()

def replace(old, new):
    global code
    if code.count(old) != 1:
        raise SystemExit(f"Unexpected STEP11H code near {old!r}")
    code = code.replace(old, new, 1)

replace(
    "train, onnx, engines, data, out, trt_site = (",
    "train, fp32, val, fp16, data, out, trt_site = (",
)
replace('sys.path.insert(0, str(onnx / "ort_packages"))\n', "")
replace("import onnxruntime as ort\n", "")
replace(
    '    with (onnx / f"{category}_onnx_val.csv").open(newline="") as stream:',
    '    with (val / f"{category}_trt_fp32_val.csv").open(newline="") as stream:',
)

# Keep the STEP11H engine and tensor-interface checks for both engines.
start = code.index('    stem = f"{category}_efficientad_s_fp32"')
engine_start = code.index("    engine = runtime.deserialize_cuda_engine(", start)
end = code.index("    score_max = map_max = 0.0", engine_start)
engine_code = code[engine_start:end]
engine_code = engine_code.replace(
    '(engines / f"{stem}.engine").read_bytes()', "engine_path.read_bytes()"
).replace("    cuda_stream = torch.cuda.current_stream()\n", "")
helper = "def open_engine(engine_path, category):\n" + engine_code
helper += "    return engine, context, buffers\n\n"
code = code[:start] + textwrap.indent(
    '''stem = f"{category}_efficientad_s"
ref_engine, ref_context, ref_buffers = open_engine(
    fp32 / f"{stem}_fp32.engine", category,
)
candidate_engine, candidate_context, candidate_buffers = open_engine(
    fp16 / f"{stem}_fp16.engine", category,
)
cuda_stream = torch.cuda.Stream()
''',
    "    ",
) + code[end:]

helper += '''def infer(context, buffers, image, cuda_stream):
    with torch.cuda.stream(cuda_stream):
        buffers["image"].copy_(image)
        if not context.execute_async_v3(cuda_stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed")
    cuda_stream.synchronize()
    return tuple(buffers[name].cpu().numpy() for name in output_names)

'''
replace("results = {}\n", helper + "results = {}\n")
start = code.index("            ort_score, ort_map = session.run")
end = code.index("            score_error, bad_score", start)
code = code[:start] + '''            ref_score, ref_map = infer(ref_context, ref_buffers, image, cuda_stream)
            trt_score, trt_map = infer(
                candidate_context, candidate_buffers, image, cuda_stream,
            )

''' + code[end:]
replace("    del context, engine\n",
        "    del candidate_context, ref_context, candidate_engine, ref_engine\n")
replace('    "onnxruntime": ort.__version__,\n', "")
replace('    "reference_provider": "CPUExecutionProvider",',
        '    "reference": "TensorRT FP32 (--noTF32)",\n'
        '    "candidate": "TensorRT FP16-enabled (--fp16 --noTF32)",\n'
        '    "cuda_stream": "non-default",')
code = (code.replace("TENSORRT FP32 VAL", "TENSORRT FP16 VS FP32 VAL")
            .replace('f"{category}_trt_fp32_val.csv").open("w"',
                     'f"{category}_trt_fp16_val.csv").open("w"')
            .replace('"ort_score", "trt_score"', '"fp32_score", "fp16_score"')
            .replace("ort_score", "ref_score")
            .replace("ort_map", "ref_map")
            .replace("TRT {actual.shape}", "FP16 candidate {actual.shape}")
            .replace("ORT {reference.shape}", "FP32 reference {reference.shape}"))

compile(code, str(target), "exec")
target.write_text(code)
