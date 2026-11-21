def open_engine(engine_path, category):
    engine = runtime.deserialize_cuda_engine(
        engine_path.read_bytes()
    )
    if engine is None:
        raise RuntimeError(f"{category}: could not deserialize engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"{category}: could not create execution context")

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    if len(names) != 3 or set(names) != {"image", *output_names}:
        raise ValueError(f"{category}: unexpected tensor names {names}")

    buffers = {}
    for name in names:
        mode = trt.TensorIOMode.INPUT if name == "image" else trt.TensorIOMode.OUTPUT
        if (engine.get_tensor_mode(name) != mode
                or engine.get_tensor_dtype(name) != trt.float32
                or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
                or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE):
            raise ValueError(f"{category}: unexpected tensor interface for {name}")
        shape = tuple(context.get_tensor_shape(name))
        if any(d <= 0 for d in shape):
            raise ValueError(f"{category}: unresolved tensor shape {name}: {shape}")
        buffers[name] = torch.empty(shape, device="cuda", dtype=torch.float32)
        if not context.set_tensor_address(name, buffers[name].data_ptr()):
            raise RuntimeError(f"{category}: buffer binding failed for {name}")

    if (tuple(buffers["image"].shape) != input_shape
            or buffers["pred_score"].numel() != 1
            or tuple(buffers["anomaly_map"].shape) != (1, 1, 256, 256)):
        raise ValueError(f"{category}: unexpected tensor shapes")

    return engine, context, buffers
