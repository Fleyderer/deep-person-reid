import argparse
import torchreid


def train_reid_model(
    exp_name,
    model_name,
    loss,
    dist_metric,
    pretrained,
    optimizer_name,
    scheduler_name,
    label_smooth,
    epochs,
    sources,
    targets,
    height,
    width,
    bs_train,
    bs_test,
    transforms,
    market1501_500k
):
    if not targets:
        targets = sources

    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources=sources,
        targets=targets,
        height=height,
        width=width,
        batch_size_train=bs_train,
        batch_size_test=bs_test,
        transforms=transforms,
        market1501_500k=market1501_500k
    )

    model = torchreid.models.build_model(
        name=model_name,
        num_classes=datamanager.num_train_pids,
        loss=loss,
        pretrained=pretrained
    )
    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim=optimizer_name,
        lr=0.0003  # You can also make this a CLI argument
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler=scheduler_name,
        stepsize=20  # Also optional for CLI
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=label_smooth
    )

    engine.run(
        save_dir=f"log/{exp_name}",
        max_epoch=epochs,
        eval_freq=10,
        print_freq=10,
        test_only=False,
        dist_metric=dist_metric
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ReID model with Torchreid")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="osnet_ain_x1_0")
    parser.add_argument("--loss", type=str, default="softmax")
    parser.add_argument("--dist-metric", type=str, default="cosine")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", type=str, default="single_step")
    parser.add_argument("--label-smooth", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--bs-train", type=int, default=32)
    parser.add_argument("--bs-test", type=int, default=100)
    parser.add_argument("--transforms", nargs="+", default=["random_flip"])
    parser.add_argument("--market1501-500k", action="store_true")

    args = parser.parse_args()

    train_reid_model(
        exp_name=args.exp_name,
        model_name=args.model_name,
        loss=args.loss,
        dist_metric=args.dist_metric,
        pretrained=args.pretrained,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        label_smooth=args.label_smooth,
        epochs=args.epochs,
        sources=args.sources,
        targets=args.targets,
        height=args.height,
        width=args.width,
        bs_train=args.bs_train,
        bs_test=args.bs_test,
        transforms=args.transforms,
        market1501_500k=args.market1501_500k
    )
