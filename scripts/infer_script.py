from Dataset import DataReader
from torch.utils.data import DataLoader
from Operator import ModelInfer
from Dataset import MSDataset
from DIAModel import DIAClipV3
from XIC import XICManager
from Args import Args, init_arg_parser
import torch
import subprocess
import os
import shutil

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    parser = init_arg_parser()
    args = Args(parser.parse_args().config)

    # prepare model and datasets
    mzml_path_list = [mzml_path.strip() for mzml_path in args.get_config('Preprocess', 'mzml_path').split(';')]
    if args.get_config('General', 'dtype', default='float') == 'float':
        model = DIAClipV3(args=args)
    else:
        model = DIAClipV3(args=args).half()
    model_infer = ModelInfer(args=args, model=model)

    for mzml_path in mzml_path_list:
        # run diann
        run_diann(args, mzml_path)

        # process xic
        xic_datas = XICManager(args=args, mzml_path=mzml_path).process()

        # form datasets
        data_reader = DataReader(args=args, xic_datas=xic_datas)
        data_info = data_reader.load_preprocess_data()
        test_dataset = MSDataset(args=args, type='test', data_info=data_info)
        test_loader = DataLoader(test_dataset, batch_size=args.get_config('Database', 'batch_size', default=1), shuffle=False, drop_last=False, collate_fn=test_dataset.collect_fn)
        logging.info(f"Loading Dataset Completed, Count: {len(test_dataset)}\n")

        model_infer.infer(data_loader=test_loader, xic_datas=xic_datas, basename=os.path.basename(mzml_path).replace('.mzML', ''))

        logging.info(
            "Inference completed. DIA-CLIP all/FDR Parquet and FDR TSV were saved under %s",
            args.get_config('General', 'out_dir'),
        )

def run_diann(args, mzml_path):
    diann_dir = args.get_config('Preprocess', 'diann_dir')
    executable_name = 'diann.exe' if os.name == 'nt' else 'diann-linux'
    diann_path = os.path.join(diann_dir, executable_name)
    if not os.path.isfile(diann_path):
        raise ValueError(f"Error: No {executable_name} found in {diann_dir}")
    library_path = args.get_config('Preprocess', 'library_path')
    threads = args.get_config('General', 'threads', default=12)
    out_dir = os.path.join(args.get_config('General', 'out_dir'), 'diann')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logging.info(f"Created output directory: {out_dir}")

    cmd = [
        diann_path,
        "--f", mzml_path,
        "--lib", library_path,
        "--threads", str(threads),
        "--out", os.path.join(out_dir, 'all_report.parquet'),
        "--out-lib", os.path.join(out_dir, 'all_lib.parquet'),
        "--qvalue", "1.0",
        "--out-lib-qvalue", "1.0",
        "--matrices",
        "--unimod4",
        "--reanalyse",
        "--rt-profiling",
        "--report-decoys",
    ]

    logging.info(f"Running DIA-NN with command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        logging.info(f"DIA-NN finished successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running DIA-NN: {e}")

    # for file in os.listdir('.'):
    #     if file.endswith('.tsv'):
    #         shutil.move(file, os.path.join(out_dir, file))
    logging.info(f"DIA-NN results saved to {out_dir}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
