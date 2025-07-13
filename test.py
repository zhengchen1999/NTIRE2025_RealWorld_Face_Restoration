import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util


def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        # CodeFormer baseline, NIPS 2022
        from models.team00_CodeFormer import main as CodeFormer
        name = f"{model_id:02}_CodeFormer_baseline"
        model_path = os.path.join('model_zoo', 'team00_CodeFormer')
        model_func = CodeFormer

    elif model_id == 1:
        from models.team01_AllForFace import main as AllForFace
        name = f"{model_id:02}_AllForFace_baseline"
        model_path = os.path.join('model_zoo', 'team01_AllForFace')
        model_func = AllForFace

    elif model_id == 2:
        from models.team02_SDFace.main import main as SDFace
        name = f"{model_id:02}_SDFace"
        model_path = os.path.join('model_zoo', 'team02_SDFace', 'net_g_final.pth')
        model_func = SDFace

    elif model_id == 3:
        from models.team03_PiSAMAP import main as PiSAMAP
        name = f"{model_id:02}_PiSAMAP"
        model_path = os.path.join('model_zoo', 'team03_PiSAMAP')
        model_func = PiSAMAP

    elif model_id == 4:
        from models.team04_MiPortraitSR.main import MPSR
        name = f"{model_id:02}_MiPortraitSR"
        model_path = os.path.join('model_zoo', 'team04_MiPortraitSR')
        model_func = MPSR

    elif model_id==5:
        name = f"{model_id:02}_faceRes"
        from models.team05_faceRes.combined_inference import run_inference
        model_path=os.path.join("model_zoo", 'team02_faceRes')
        model_func=run_inference

    elif model_id==500:
        name = f"{model_id:02}_ZSSR"
        from models.team05_ZSSR.zssr import ZSSRWrapper
        from models.team05_ZSSR.config import set_config
        config = set_config()
        config.accelerator = "gpu" if device.type == "cuda" else "cpu"
        zssr_wrapper = ZSSRWrapper(config)
        model_path = None
        model_func = zssr_wrapper.wrapper

    elif model_id == 6:     #change the model_id with Model ID
        from models.team06_DSS import main as Dssmodel
        name = f"{model_id:02}_Dssmodel_baseline"
        model_path = os.path.join('model_zoo', 'team06_DSS')      #abs path
        model_func = Dssmodel

    elif model_id == 7:
        from models.team07_DiffBIR.inference_diffbir import main as team07_DiffBIR
        name = f"{model_id:02}_DiffBIR"
        model_path = os.path.join('model_zoo', 'team07_DiffBIR')
        model_func = team07_DiffBIR

    elif model_id == 8:
        from models.team08_DCMoE import main as DCMoE
        name = f"{model_id:02}_DCMoE"
        model_path = os.path.join('model_zoo', 'team08_DCMoE')
        model_func = DCMoE

    elif model_id == 9:
        from models.team09_good import main as good
        name = f"{model_id:02}_good"
        model_path = os.path.join('model_zoo', 'team09_good')
        model_func = good

    elif model_id == 10:
        # CodeFormer baseline, NIPS 2022
        from models.team10_cfDiffbir.pipeline import pipe as team10_pipeline
        name = f"{model_id:02}_cfDiffbir"
        model_path = os.path.join('model_zoo', 'team10_cfDiffbir')
        model_func = team10_pipeline

    elif model_id == 12:
        from models.team12_diffbir.run import run as run_infer
        name = f"{model_id:02}_CustomModel"
        model_path = None
        model_func = run_infer

    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    return model_func, model_path, name


def run(model_func, model_name, model_path, device, args, mode="test"):
    # --------------------------------
    # dataset path
    # --------------------------------
    if mode == "valid":
        data_path = args.valid_dir
    elif mode == "test":
        data_path = args.test_dir
    assert data_path is not None, "Please specify the dataset path for validation or test."
    
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    data_paths = []
    save_paths = []
    for dataset_name in ("CelebA", "Wider-Test", "LFW-Test", "WebPhoto-Test", "CelebChild-Test"):
        data_paths.append(os.path.join(data_path, dataset_name))
        save_paths.append(os.path.join(save_path, dataset_name))
        util.mkdir(save_paths[-1])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for data_path, save_path in zip(data_paths, save_paths):
        model_func(model_dir=model_path, input_path=data_path, output_path=save_path, device=device)
    end.record()
    torch.cuda.synchronize()
    print(f"Model {model_name} runtime (Including I/O): {start.elapsed_time(end)} ms")


def main(args):

    utils_logger.logger_info("NTIRE2025-RealWorld-Face-Restoration", log_path="NTIRE2025-RealWorld-Face-Restoration.log")
    logger = logging.getLogger("NTIRE2025-RealWorld-Face-Restoration")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model_func, model_path, model_name = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if args.valid_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="valid")
        
    if args.test_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2025-RealWorld-Face-Restoration")
    parser.add_argument("--valid_dir", default=None, type=str, help="Path to the validation set")
    parser.add_argument("--test_dir", default=None, type=str, help="Path to the test set")
    parser.add_argument("--save_dir", default="NTIRE2025-RealWorld-Face-Restoration/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)

    args = parser.parse_args()
    pprint(args)

    main(args)
