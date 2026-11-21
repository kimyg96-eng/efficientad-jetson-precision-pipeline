import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
code = source.read_text()

def replace_block(start, end, replacement):
    global code
    if code.count(start) != 1 or code.count(end) != 1:
        raise SystemExit(f"Unexpected STEP11D-R1 script near {start!r}")
    left = code.index(start)
    right = code.index(end, left)
    code = code[:left] + replacement + code[right:]

code = code.replace(
    "import tarfile\n", "import tarfile\nimport time\n"
).replace(
    "from torchvision import transforms\n",
    "from torchvision import transforms\nfrom torchvision.datasets import ImageFolder\n",
)

replace_block(
    'with tarfile.open(assets / "imagenette2.tgz"',
    "normal_transform = ",
    '''steps = 40000
labels_root = out / "imagenette"
with tarfile.open(assets / "imagenette2.tgz", "r|gz") as archive:
    for member in archive:
        parts = PurePosixPath(member.name).parts
        if (member.isfile() and len(parts) == 4 and parts[1] == "train"
                and PurePosixPath(member.name).suffix.lower() in {".jpeg", ".jpg"}):
            if any(part in {"", ".", "..", "/"} for part in parts):
                raise ValueError(f"Invalid ImageNette archive path: {member.name}")
            archive.extract(member, path=labels_root, filter="data")

roots = sorted(path for path in labels_root.glob("*/train") if path.is_dir())
if len(roots) != 1:
    raise SystemExit("Expected one extracted ImageNette train directory")

''',
)
code = code.replace(
    "auxiliary = tensor_from_bytes(auxiliary_bytes, auxiliary_transform).cuda()",
    '''auxiliary_dataset = ImageFolder(roots[0], transform=auxiliary_transform)
if len(auxiliary_dataset.classes) != 10 or not auxiliary_dataset.samples:
    raise ValueError("Unexpected ImageNette class inventory")
print(f"IMAGENETTE_TRAIN_IMAGES={len(auxiliary_dataset)}", flush=True)''',
)
replace_block(
    '    first_train = read_normal(',
    "    stats = module.teacher_channel_mean_std(",
    '''    train_tensors = [read_normal(row, category, "TRAIN") for row in train]
    val_tensors = [read_normal(row, category, "VAL") for row in val]

    def teacher_batches():
        for tensor in train_tensors:
            yield SimpleNamespace(image=tensor)

''',
)
replace_block(
    "    before = params[0].detach().clone()",
    'with (out / "input_sha256.csv")',
    '''    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * steps), gamma=0.1,
    )

    def shuffled_indices(size, generator):
        while True:
            yield from torch.randperm(size, generator=generator).tolist()

    train_indices = shuffled_indices(
        len(train_tensors), torch.Generator().manual_seed(seed),
    )
    auxiliary_indices = shuffled_indices(
        len(auxiliary_dataset), torch.Generator().manual_seed(seed + 1),
    )

    started = time.monotonic()
    with (out / f"{category}_loss.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "loss_st", "loss_ae", "loss_stae", "total", "lr"])
        for step in range(1, steps + 1):
            x = train_tensors[next(train_indices)].cuda()
            aux, _ = auxiliary_dataset[next(auxiliary_indices)]
            aux = aux.unsqueeze(0).cuda()
            lr = optimizer.param_groups[0]["lr"]

            optimizer.zero_grad(set_to_none=True)
            losses = model(batch=x, batch_imagenet=aux)
            total = sum(losses)
            values = torch.stack([*losses, total]).detach().tolist()
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{category}: non-finite loss at step {step}")
            total.backward()
            optimizer.step()
            scheduler.step()
            writer.writerow([step, *values, lr])

            if step == 1 or step % 500 == 0:
                stream.flush()
                print(
                    f"{category}: STEP={step}/{steps} LOSS={values[-1]:.7g} "
                    f"ELAPSED_S={time.monotonic() - started:.1f}",
                    flush=True,
                )

    if any(not torch.isfinite(param).all().item() for param in params):
        raise ValueError(f"{category}: non-finite learned parameters")

    model.eval()
    val_batches = (
        SimpleNamespace(image=tensor, gt_label=torch.zeros(1, dtype=torch.long))
        for tensor in val_tensors
    )
    quantiles = module.map_norm_quantiles(val_batches)
    if (any(not torch.isfinite(v).all().item() for v in quantiles.values())
            or any(quantiles[f"qb_{key}"].item() <= quantiles[f"qa_{key}"].item()
                   for key in ("st", "ae"))):
        raise ValueError(f"{category}: invalid VAL map quantiles")
    model.quantiles.update(quantiles)

    with (out / f"{category}_val_scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "score"])
        with torch.inference_mode():
            for row, tensor in zip(val, val_tensors, strict=True):
                pred = model(tensor.cuda())
                if (pred.pred_score.numel() != 1
                        or tuple(pred.anomaly_map.shape) != (1, 1, 256, 256)
                        or not torch.isfinite(pred.pred_score).all().item()
                        or not torch.isfinite(pred.anomaly_map).all().item()):
                    raise ValueError(f"{category}: invalid VAL output for {row['path']}")
                writer.writerow([row["path"], pred.pred_score.item()])

    checkpoint = out / f"{category}_efficientad_s_fp32.pt"
    partial = checkpoint.with_suffix(".partial")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, partial)
    partial.replace(checkpoint)

    results[category] = {
        "train_images": len(train),
        "val_images": len(val),
        "optimizer_steps": steps,
        "map_quantiles": {key: value.item() for key, value in quantiles.items()},
        "checkpoint": checkpoint.name,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(f"{category}: FP32_TRAIN=PASS / NORMAL_VAL_CALIBRATION=PASS", flush=True)

''',
)
code = code.replace("import json\n", "import json\nimport math\n")
code = code.replace("one-step FP32 training pipeline pilot", "FP32 training and normal VAL map calibration")
code = code.replace(
    '    "imagenette_member": auxiliary_name,\n'
    '    "imagenette_sha256": digest(auxiliary_bytes),\n',
    '    "imagenette_train_images": len(auxiliary_dataset),\n'
    '    "imagenette_classes": auxiliary_dataset.classes,\n'
    '    "batch_size": 1,\n'
    '    "steps_per_category": steps,\n'
    '    "learning_rate": 1e-4,\n'
    '    "weight_decay": 1e-5,\n'
    '    "lr_decay_step": int(0.95 * steps),\n',
)
compile(code, str(target), "exec")
target.write_text(code)
