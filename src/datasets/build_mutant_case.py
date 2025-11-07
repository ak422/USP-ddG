import argparse
import posixpath
import subprocess
import requests
import tarfile
import tempfile
import pandas as pd
import numpy as np
import os
import logging
from functools import partial
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

from fix_pdb import fix_pdb

PDB_TO_FIX = ['1C4Z', '2NYY', '2NZ9', '3VR6', '4GNK', '4GXU', '4K71', '4NM8']

def download_file(url, destination_path):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    else:
        print(f"Failed to download file. HTTP response status code: {response.status_code}")

def extract_tar_gz(file_path, destination_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=destination_path, filter="data")

def run_foldx_repair(file: Path, timeout: float = 9600, output_dir: Path = Path("repaired_pdbs")):
    command1 = f"./FoldX --command RepairPDB --ionStrength 0.15 --pdb-dir {file.parent} --output-dir {output_dir} --pdb {file.name}"
    command2 = f'find . -type f -name {output_dir}/{file.name} -delete'
    command3 = f"rename 's/_Repair//' {output_dir}/*.pdb"
    result1 = subprocess.run(command1, shell=True, check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             timeout=timeout)
    result2 = subprocess.run(command2, shell=True, check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             timeout=timeout)
    result3 = subprocess.run(command3, shell=True, check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             timeout=timeout)
    return (file, None) if result3.returncode == 0 else (file, result3.stderr)


def run_foldx_buildmodel_mt(mutant_file: Path, pdb_dir: Path, foldx_dir: Path, output_wt_dir: Path = Path("wild_type"), output_mt_dir: Path = Path("mutant")):
    pdbcode = mutant_file.name.split('_')[2] + '_' + mutant_file.name.split('_')[3]
    complex = mutant_file.name.split('_')[3]
    pdb = complex + '.pdb'
    work_dir = Path(foldx_dir.parent, pdbcode)
    work_dir.mkdir(parents=True, exist_ok=True)
    command1 = f'./FoldX --command=BuildModel --numberOfRuns=1 --pdb={pdb}  --mutant-file={mutant_file}  --output-dir={work_dir} --pdb-dir={pdb_dir} >{work_dir}/foldx.log'
    command2 = f'mv {work_dir}/{complex}_1.pdb {output_mt_dir}/{pdbcode}.pdb'
    command3 = f'mv {work_dir}/WT_{complex}_1.pdb {output_wt_dir}/{pdbcode}.pdb'
    # foldx_ddg
    command4 = f"mv {work_dir}/Dif_{complex}.fxout  {foldx_dir}/{pdbcode}.txt"
    command5 = f'rm -r {work_dir}'
    # Use subprocess to execute the command in the shell
    process1 = subprocess.run(command1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process2 = subprocess.run(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process3 = subprocess.run(command3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process4 = subprocess.run(command4, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process5 = subprocess.run(command5, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Return a tuple of the file name and the stdout or stderr if command fails
    return (mutant_file, None) if process5.returncode == 0 else (mutant_file, process5.stderr)

def run_foldx_optimize(file: Path, pdb_dir: Path, output_dir: Path = Path("wild_type")):
    pdbcode = file.name.split('.')[0]
    work_dir = Path(output_dir.parent, pdbcode)
    work_dir.mkdir(parents=True, exist_ok=True)
    command1 = f'./FoldX --command=Optimize --pdb={file.name}  --output-dir={work_dir} --pdb-dir={pdb_dir} >{work_dir}/foldx.log'
    command2 = f'mv {work_dir}/Optimized_{file.name} {output_dir}/{file.name}'
    command3 = f'rm -r {work_dir}'
    # Use subprocess to execute the command in the shell
    process1 = subprocess.run(command1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process2 = subprocess.run(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process3 = subprocess.run(command3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Return a tuple of the file name and the stdout or stderr if command fails
    return (file, None) if process3.returncode == 0 else (file, process3.stderr)

def run_foldx_analysecomplex_wt(file: Path, pdb_dir: Path, partner, output_dir: Path = Path("wild_type")):
    pdbcode = file.name.split('.')[0]
    complex = pdbcode.split('_')[1]
    work_dir = Path(output_dir.parent, pdbcode)
    work_dir.mkdir(parents=True, exist_ok=True)
    if complex not in partner.keys():
        return (file, -1)
    command1 = f"./FoldX --command=AnalyseComplex --pdb={file.name} --analyseComplexChains={','.join(partner[complex][0])} --output-dir={work_dir} --pdb-dir={pdb_dir} >{work_dir}/foldx.log"
    command2 = f'mv {work_dir}/Interaction_{pdbcode}_AC.fxout {output_dir}/Interaction_{pdbcode}.txt'
    command3 = f'rm -r {work_dir}'
    # Use subprocess to execute the command in the shell
    try:
        process1 = subprocess.run(command1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process2 = subprocess.run(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process3 = subprocess.run(command3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return (file, None) if process3.returncode == 0 else (file, process3.stderr)
    except subprocess.CalledProcessError as e:
        # Handle errors in the called executable
        return (file, e.stderr)
    except Exception as e:
        # Handle other exceptions such as file not found or permissions issues
        return (file, str(e).encode())

def run_foldx_analysecomplex_mt(file: Path, pdb_dir: Path, partner, output_dir: Path = Path("wild_type")):
    pdbcode = file.name.split('.')[0]
    complex = pdbcode.split('_')[1]
    work_dir = Path(output_dir.parent, pdbcode)
    work_dir.mkdir(parents=True, exist_ok=True)
    if complex not in partner.keys():
        return (file, -1)
    command1 = f"./FoldX --command=AnalyseComplex --pdb={file.name} --analyseComplexChains={','.join(partner[complex][0])} --output-dir={work_dir} --pdb-dir={pdb_dir} >{work_dir}/foldx.log"
    command2 = f'mv {work_dir}/Interaction_{pdbcode}_AC.fxout {output_dir}/Interaction_{pdbcode}.txt'
    command3 = f'rm -r {work_dir}'
    # Use subprocess to execute the command in the shell
    try:
        process1 = subprocess.run(command1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process2 = subprocess.run(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process3 = subprocess.run(command3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return (file, None) if process3.returncode == 0 else (file, process3.stderr)
    except subprocess.CalledProcessError as e:
        # Handle errors in the called executable
        return (file, e.stderr)
    except Exception as e:
        # Handle other exceptions such as file not found or permissions issues
        return (file, str(e).encode())

def run_foldx_analysecomplex_op(file: Path, pdb_dir: Path, partner, output_dir: Path = Path("wild_type")):
    pdbcode = file.name.split('.')[0]
    complex = pdbcode.split('_')[1]
    work_dir = Path(output_dir.parent, pdbcode)
    work_dir.mkdir(parents=True, exist_ok=True)
    if complex not in partner.keys():
        return (file, -1)
    command1 = f"./FoldX --command=AnalyseComplex --pdb={file.name} --analyseComplexChains={','.join(partner[complex][0])} --output-dir={work_dir} --pdb-dir={pdb_dir} >{work_dir}/foldx.log"
    command2 = f'mv {work_dir}/Interaction_{pdbcode}_AC.fxout {output_dir}/Interaction_{pdbcode}.txt'
    command3 = f'rm -r {work_dir}'
    # Use subprocess to execute the command in the shell
    try:
        process1 = subprocess.run(command1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process2 = subprocess.run(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process3 = subprocess.run(command3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return (file, None) if process3.returncode == 0 else (file, process3.stderr)
    except subprocess.CalledProcessError as e:
        # Handle errors in the called executable
        return (file, e.stderr)
    except Exception as e:
        # Handle other exceptions such as file not found or permissions issues
        return (file, str(e).encode())

def create_individual_file(csv_path, output_individual_dir):
    processed_skempi = pd.read_csv(csv_path, sep=',')
    mutant_file_wt = []
    mutant_file_mt = []
    for i, row in processed_skempi.iterrows():
        pdbcode = row['#Pdb']
        mut_list = row["Mutation(s)_cleaned"].split(",")
        wild_list = []
        for mut in mut_list:
            wildname = list(mut)[0]
            chainid = list(mut)[1]
            resid = "".join(list(mut)[2:-1])
            mutname = list(mut)[-1]
            wild_list.append("".join([wildname, chainid, resid, wildname]))
        wildstr = ",".join(wild_list) + ";"
        mutstr = ",".join(mut_list) + ";"
        mut_file_wt = Path(output_individual_dir, 'individual_list_' + pdbcode + '_wt.txt')
        mut_file_mt = Path(output_individual_dir, 'individual_list_' + pdbcode + '_mt.txt')
        mutant_file_wt.append(mut_file_wt)
        mutant_file_mt.append(mut_file_mt)
        with open(mut_file_wt, 'w') as f:
            cont = wildstr
            f.write(cont)
        with open(mut_file_mt, 'w') as f:
            cont = mutstr
            f.write(cont)
    return mutant_file_wt, mutant_file_mt

def get_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter('[%(asctime)s] %(message)s',"%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger

def func1(pdbcode, path="optimized", subset='HER2'):
    prefix_dir = f'../../data/SKEMPI2/{subset}_cache'
    Interaction_mt = f"{prefix_dir}/mutant_ddg/Interaction_{pdbcode}.txt"
    # Interaction_op = f"{prefix_dir}/optimize_ddg/Interaction_{pdbcode}.txt"
    Interaction_wt = f"{prefix_dir}/wildtype_ddg/Interaction_{pdbcode}.txt"

    try:
        with open(Interaction_mt,"r") as file:
            lines = file.readlines()
            for line in lines:
                if pdbcode in line:
                    Inter_mt = float(line.split()[5])
        # with open(Interaction_op,"r") as file:
        #     lines = file.readlines()
        #     for line in lines:
        #         if pdbcode in line:
        #             Inter_op = float(line.split()[5])
        with open(Interaction_wt,"r") as file:
            lines = file.readlines()
            for line in lines:
                if pdbcode in line:
                    Inter_wt = float(line.split()[5])

        label = Inter_mt - Inter_wt
    except:
        label = None
    return label

if __name__ == '__main__':
    # -HER2-
    MAX_CPU_cores = 32
    # subset: python build_mutant_case.py --subset CR6261
    parser0 = argparse.ArgumentParser("First parser for initial arguments")
    parser0.add_argument('--subset', type=str, default=f'CR6261')
    # 解析已知参数并获取剩下的参数
    args0, remaining_args = parser0.parse_known_args()
    subset = args0.subset

    parser = argparse.ArgumentParser(description="Second parser for remaining arguments")
    prefix = Path(f'../../data/SKEMPI2/{subset}_cache')
    parser.add_argument('--csv-path', type=Path, default=Path(f'{prefix.parent}/{subset}.csv'), help=f'Path to the {subset} CSV file')
    parser.add_argument('--output-csv-path', type=Path, default=Path(f'{prefix}/{subset}.csv'), help=f'Path to the processed {subset} CSV file')
    parser.add_argument('--pdb-dir', type=Path, default=Path(f'{prefix}/PDBs'), help=f'Path to the {subset} PDB directory, must have name "PDBs"')
    parser.add_argument('--output-pdb-dir', type=Path, default=Path(f'{prefix}/processed_pdbs'), help='Path to repaired PDBs')
    parser.add_argument('--output-individual-dir', type=Path, default=Path(f'{prefix}/individual_file'),
                        help='Path to individual file')
    parser.add_argument('--output-wt-dir', type=Path, default=Path(f'{prefix}/wildtype'), help='Path to wild-type PDBs')
    parser.add_argument('--output-mt-dir', type=Path, default=Path(f'{prefix}/mutant'), help='Path to mutant PDBs')
    parser.add_argument('--output-foldx-dir', type=Path, default=Path(f'{prefix}/FoldX_ddg'), help='Path to FoldX_ddg')
    parser.add_argument('--output-optimized-dir', type=Path, default=Path(f'{prefix}/optimized'), help='Path to optimized PDBs')
    parser.add_argument('--output-analysecomplex-wt-dir', type=Path, default=Path(f'{prefix}/wildtype_ddg'), help='Path to analysecomplex Txts')
    parser.add_argument('--output-analysecomplex-mt-dir', type=Path, default=Path(f'{prefix}/mutant_ddg'), help='Path to analysecomplex Txts')
    parser.add_argument('--output-analysecomplex-op-dir', type=Path, default=Path(f'{prefix}/optimize_ddg'), help='Path to analysecomplex Txts')
    parser.add_argument('--no-repair', action='store_true', help='skip FoldX RepairPDB step')
    args = parser.parse_args(remaining_args)
    logger = get_logger()
    logger.info(f'Generating {subset} ...')

    if args.no_repair:
        print("Skipping repair step.")
        exit()

    args.pdb_dir.mkdir(parents=True, exist_ok=True)
    args.output_pdb_dir.mkdir(parents=True, exist_ok=True)
    args.output_individual_dir.mkdir(parents=True, exist_ok=True)
    args.output_wt_dir.mkdir(parents=True, exist_ok=True)
    args.output_mt_dir.mkdir(parents=True, exist_ok=True)
    args.output_foldx_dir.mkdir(parents=True, exist_ok=True)
    args.output_optimized_dir.mkdir(parents=True, exist_ok=True)
    args.output_analysecomplex_wt_dir.mkdir(parents=True, exist_ok=True)
    args.output_analysecomplex_mt_dir.mkdir(parents=True, exist_ok=True)
    args.output_analysecomplex_op_dir.mkdir(parents=True, exist_ok=True)

    # copy PDBs
    subprocess.run(f'cp -r {prefix.parent}/{subset}_PDBs/*  {args.pdb_dir}', shell=True)

    # run FoldX repair
    pdb_fpaths = [p for p in args.pdb_dir.glob('*.pdb') if not p.name.startswith('.')]
    _run_foldx_repair = partial(run_foldx_repair, output_dir=args.output_pdb_dir)
    with mp.Pool(min(MAX_CPU_cores, mp.cpu_count())) as pool:
        result = pool.map(_run_foldx_repair, tqdm(pdb_fpaths, desc=f'\033[0;37;42m Repair for PDBs\033[0m'))
    success_count = sum(r[1] is None for r in result)
    print(f"{success_count} PDBs Repair by FoldX")
    if pdb_fpaths != []:
        subprocess.run(f"rm {args.output_pdb_dir}/*.fxout", shell=True)

    # buildmodel
    mutant_file_wt, mutant_file_mt = create_individual_file(args.csv_path, args.output_individual_dir)
    _run_foldx_buildmodel_mt = partial(run_foldx_buildmodel_mt, pdb_dir=args.output_pdb_dir,
                                       foldx_dir=args.output_foldx_dir, output_wt_dir=args.output_wt_dir,
                                       output_mt_dir=args.output_mt_dir)
    with mp.Pool(min(MAX_CPU_cores, mp.cpu_count())) as pool:
        result = pool.map(_run_foldx_buildmodel_mt,
                          tqdm(mutant_file_mt, desc=f'\033[0;37;42m Buildmodel for mutant\033[0m'))
    success_count = sum(r[1] is None for r in result)
    print(f"{success_count} PDBs of mutant Buildmodel by FoldX")

    # AnalyseComplex
    df = pd.read_csv(args.csv_path, sep=',')
    complex_partner_dict = df.groupby('complex').apply(lambda x: list(zip(x['Partner1'], x['Partner2'])), include_groups=False).to_dict()
    complex_partner = {key: list(set(value)) for key, value in complex_partner_dict.items()}

    # wild-type
    wildtype_fpaths = [p for p in args.output_wt_dir.glob('*.pdb')]
    _run_foldx_analysecomplex_wt = partial(run_foldx_analysecomplex_wt, pdb_dir=args.output_wt_dir,
                                           partner=complex_partner, output_dir=args.output_analysecomplex_wt_dir)
    with mp.Pool(min(MAX_CPU_cores, mp.cpu_count())) as pool:
        result = pool.map(_run_foldx_analysecomplex_wt, tqdm(wildtype_fpaths, desc=f'\033[0;37;42m AnalyseComplex for wild-type\033[0m'))
    success_count = sum(r[1] is None for r in result)
    print(f"{success_count} PDBs of wild-type AnalyseComplex by FoldX")

    # mutant
    mutant_fpaths = [p for p in args.output_mt_dir.glob('*.pdb') if not p.name.startswith('.')]
    _run_foldx_analysecomplex_mt = partial(run_foldx_analysecomplex_mt, pdb_dir=args.output_mt_dir,
                                           partner=complex_partner, output_dir=args.output_analysecomplex_mt_dir)
    with mp.Pool(min(MAX_CPU_cores, mp.cpu_count())) as pool:
        result = pool.map(_run_foldx_analysecomplex_mt, tqdm(mutant_fpaths, desc=f'\033[0;37;42m AnalyseComplex for mutant\033[0m'))
    success_count = sum(r[1] is None for r in result)
    print(f"{success_count} PDBs of mutant AnalyseComplex by FoldX")

    # energy optimize
    optimize_fpaths = [p for p in args.output_mt_dir.glob('*.pdb') if not p.name.startswith('.')]
    _run_foldx_optimize = partial(run_foldx_optimize, pdb_dir=args.output_mt_dir,
                                  output_dir=args.output_optimized_dir)
    with mp.Pool(min(MAX_CPU_cores, mp.cpu_count())) as pool:
        result = pool.map(_run_foldx_optimize, tqdm(optimize_fpaths, desc=f'\033[0;37;42m optimize for mutant\033[0m'))
    success_count = sum(r[1] is None for r in result)
    print(f"{success_count} PDBs of mutant Optimize by FoldX")

    # update FoldX ddg
    df = pd.read_csv(args.csv_path, sep=',')
    df["ddG_FoldX"] = df.apply(lambda x: func1(x['#Pdb'], subset=subset), axis=1)  # Interaction
    logger.info(f'Writing to {prefix}/{subset}.csv')
    df.to_csv(f"{args.output_csv_path}", index=None)
    pearson = df[['ddG', 'ddG_FoldX']].corr('pearson').iloc[0, 1]
    spearman = df[['ddG', 'ddG_FoldX']].corr('spearman').iloc[0, 1]
    logger.info(f'[FoldX-ddg]: pearson={pearson:.4f}, spearman={spearman:.4f}')
