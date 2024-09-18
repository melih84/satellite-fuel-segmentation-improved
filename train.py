import argparse
import yaml
from pathlib import Path

from tensorboardX import SummaryWriter

from src.networks import unet

def train(hyp, opt, tb_writer=None):
    epochs, batch_size, device, save_dir, img_size = opt.epochs, opt.batch_size, opt.device, Path(opt.save_dir), opt.image_size

    # Save run settings
    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.dump(opt, f, sort_keys=False)

    # TODO Initialize seeds

    # Data
    with open(opt.data, "r") as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    nc = int(data_dict["nc"])
    names = data_dict["names"]
    assert len(names) == nc, f"{len(names)} names found for nc={nc} dataset in {opt.data}"
    train_path = data_dict["train"]

    # Model
    model = unet(input_size=(img_size, img_size, 3), output_classes=nc)
    model.summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    # parser.add_argument("config", type=str, help="model.yaml path")
    parser.add_argument("--hyp", type=str, default="hyperparameters/hyp.yaml")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default="runs/train")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--save-period", type=int, default=-1, help="Log model after every 'save_period' epoch")

    opt = parser.parse_args()
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)

    # Train
    tb_writer = SummaryWriter()
    train(hyp, opt, tb_writer)


    # parser.add_argument("--n_epochs", type=int)
    # parser.add_argument("--batch_size", default=8, type=int)
    # parser.add_argument("--image_size", default=None, type=int)
    # parser.add_argument("--study-id", default="study-00", type=str)
    # parser.add_argument("--seed", default=42, type=int)

    # configs = parser.parse_args()
    
    # root_path = "experiments/"
    # study_id = configs.study_id
    # study_dir = Path("./experiments") / study_id

    # if not study_dir.exists():
    #     run_num = 1
    # else:
    #     exst_run_nums = [int(str(folder.name).split('run-')[1]) for folder in
    #                      study_dir.iterdir() if
    #                      str(folder.name).startswith('run-')]
    #     if len(exst_run_nums) == 0:
    #         run_num = 1
    #     else:
    #         run_num = max(exst_run_nums) + 1
    
    # run_dir = study_dir / f"run-{run_num:02}"
    # os.makedirs(run_dir)

    # with open(run_dir / "configs.json", "w") as f:
    #     json.dump(vars(configs), f)

    # configs.run_dir = run_dir

    # print("*"*40)
    # print(f"STUDY-ID: {study_id} / RUN: {run_num}")
    # print("*"*40)

    # main(configs)