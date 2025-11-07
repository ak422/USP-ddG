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


def process_csv(df, block_list={'1KBH'}):
    # df['dG_wt'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
    # df['dG_mut'] =  (8.314/4184)*(273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])

    df['Temperature'] = df['Temperature'].apply(lambda x: float(x[:3]) if isinstance(x, str) else (273.15 + 25.0))
    df['dG_wt'] = (8.314 / 4184) * df['Temperature'] * np.log(df['Affinity_wt_parsed'])
    df['dG_mut'] = (8.314 / 4184) * df['Temperature'] * np.log(df['Affinity_mut_parsed'])

    df['ddG'] = df['dG_mut'] - df['dG_wt']

    dfs = []
    tmp_dict = {}
    for i, row in df.iterrows():
        complex, group1, group2 = row['#Pdb'].split('_')
        if complex.upper() in block_list:
            continue

        if not np.isfinite(row['ddG']):
            continue

        tmp_dict['#Pdb'] = str(i) + '_' + complex.upper()
        tmp_dict['complex'] = complex.upper()
        tmp_dict['Partner1'] = group1
        tmp_dict['Partner2'] = group2

        Partners = [group1, group2]
        Partners.sort()
        complex_PPI = complex + '_' + Partners[0] + '_' + Partners[1]
        tmp_dict['complex_PPI'] = complex_PPI

        tmp_dict['Mutation(s)_cleaned'] = row['Mutation(s)_cleaned']
        tmp_dict['ddG'] = row['ddG']
        tmp_dict['Label'] = 'forward'
        tmp_dict['Protein 1'] = row['Protein 1']
        tmp_dict['Protein 2'] = row['Protein 2']

        dfs.append(tmp_dict)
        tmp_dict = {}
        i = i + 1
    processed_df = pd.DataFrame(dfs)
    print(f"Preprocessing complete. {len(processed_df)} entries remain in the processed dataset.")

    return processed_df


def process_pdbs(pdb_dir, PDB_List):
    pdb_dir = Path(pdb_dir)

    # fix 1KBH. It stores all states in a single state, causing unresolvable clash.
    # with open(pdb_dir / '1KBH.pdb') as f:
    #     lines = f.readlines()
    # new_lines = lines[13471:14181] + lines[29285:]
    # with open(pdb_dir / '1KBH.pdb', 'w') as f:
    #     f.write('\n'.join(new_lines) + '\n')
    # print("Fixed 1KBH")

    # fix 4BFI
    with open(pdb_dir / '4BFI.pdb') as f:
        lines = f.readlines()
    new_lines = lines[:3062] + lines[3069:]
    with open(pdb_dir / '4BFI.pdb', 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
    print("Fixed 4BFI")
    #
    # fix PDB_TO_FIX
    for pdb_code in list(set(PDB_TO_FIX) & set(PDB_List)):
        fpath = pdb_dir / f'{pdb_code}.pdb'
        fix_pdb(fpath, {}, output_file=fpath)
        print(f'Fixed {pdb_code}')

def run_foldx_repair(file: Path, timeout: float = 9600, output_dir: Path = Path("repaired_pdbs")):
    command1 = f"./FoldX --command RepairPDB --ionStrength 0.15 --pdb-dir {file.parent} --output-dir {output_dir} --pdb {file.name}"
    command2 = f"rename 's/_Repair//' {output_dir}/*.pdb"
    result1 = subprocess.run(command1, shell=True, check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             timeout=timeout)
    result2 = subprocess.run(command2, shell=True, check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             timeout=timeout)
    return (file, None) if result2.returncode == 0 else (file, result2.stderr)


def run_foldx_buildmodel_mt(mutant_file: Path, pdb_dir: Path, foldx_dir: Path, output_wt_dir: Path = Path("wild_type"), output_mt_dir: Path = Path("mutant")):
    pdbcode = mutant_file.name.split('_')[2] + '_' + mutant_file.name.split('_')[3]
    complex = mutant_file.name.split('_')[3]
    pdb = complex + '.pdb'
    work_dir = Path(foldx_dir.parent, pdbcode)
    work_dir.mkdir(parents=True, exist_ok=True)
    command1 = f'./FoldX --command=BuildModel --numberOfRuns=1 --pdb={pdb}  --mutant-file={mutant_file}  --output-dir={work_dir} --pdb-dir={pdb_dir} >{work_dir}/foldx.log'
    command2 = f'mv {work_dir}/{complex}_1.pdb {output_mt_dir}/{pdbcode}.pdb'           # mutant
    command3 = f'mv {work_dir}/WT_{complex}_1.pdb {output_wt_dir}/{pdbcode}.pdb'        # wild_type
    # command3 = f'cp {pdb_dir}/{complex}.pdb {output_wt_dir}/{pdbcode}.pdb'            # wild_type
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
        if not np.isfinite(row['ddG']):
            continue
        if row['complex'] in {'1KBH'}:
            continue

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

def func1(pdbcode, path="optimized", subset='skempi_v2'):
    prefix_dir = f'../../data/SKEMPI2/{subset}_cache'
    Interaction_mt = f"{prefix_dir}/mutant_ddg/Interaction_{pdbcode}.txt"
    Interaction_wt = f"{prefix_dir}/wildtype_ddg/Interaction_{pdbcode}.txt"
    try:
        with open(Interaction_mt,"r") as file:
            lines = file.readlines()
            for line in lines:
                if pdbcode in line:
                    Inter_mt = float(line.split()[5])
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
    # -6706-
    MAX_CPU_cores = 32
    parser = argparse.ArgumentParser()
    subset = 'skempi_v2'
    prefix = Path(f'../../data/SKEMPI2/{subset}_cache')
    parser.add_argument('--csv-path', type=Path, default=Path(f'{prefix.parent}/{subset}.csv'), help='Path to the SKEMPI CSV file')
    parser.add_argument('--csv-with-all-results', type=Path, default=Path(f'{prefix.parent}/{subset}_with_all_results.csv'),
                       help='Path to the SKEMPI with all results file')
    parser.add_argument('--output-csv-path', type=Path, default=Path(f'{prefix}/{subset}.csv'), help='Path to the processed SKEMPI CSV file')
    parser.add_argument('--pdb-dir', type=Path, default=Path(f'{prefix}/PDBs'), help='Path to the SKEMPI PDB directory, must have name "PDBs"')
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
    args = parser.parse_args()
    logger = get_logger()
    logger.info(f'Generating {subset} ...')

    if not args.csv_with_all_results.exists():
        args.csv_path.parent.mkdir(exist_ok=True, parents=True)
        download_file('https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv', args.csv_path)
        df = pd.read_csv(args.csv_path, sep=';')
        processed_df = process_csv(df)
    else:
        df = pd.read_csv(args.csv_with_all_results, sep=',')
        dfs = []
        for i, row in df.iterrows():
            complex = row['complex']
            if complex.upper() in {'1KBH'}:
                continue

            if not np.isfinite(row['ddG']):
                continue
            dfs.append(row)
        processed_df = pd.DataFrame(dfs)
        processed_df.replace("1.00E+96", "1E96", inplace=True)
        processed_df.replace("1.00E+50", "1E50", inplace=True)
        print(f"Preprocessing complete. {len(processed_df)} entries remain in the processed dataset.")

    assert args.pdb_dir.name == 'PDBs'
    if not args.pdb_dir.exists():
        args.pdb_dir.parent.mkdir(exist_ok=True, parents=True)
        temp_fpath = Path(tempfile.gettempdir()) / 'SKEMPI2_PDBs.tgz'
        download_file('https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz', temp_fpath)
        extract_tar_gz(temp_fpath, args.pdb_dir.parent)

    processed_df.to_csv(args.output_csv_path, index=False)

    PDB_List = list(set(processed_df['complex']))

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

    process_pdbs(args.pdb_dir, PDB_List)

    if args.no_repair:
        print("Skipping repair step.")
        exit()

    # run FoldX repair
    pdb_fpaths = [p for p in args.pdb_dir.glob('*.pdb') if p.name.split('.')[0] in PDB_List]
    _run_foldx_repair = partial(run_foldx_repair, output_dir=args.output_pdb_dir)
    with mp.Pool(min(MAX_CPU_cores, mp.cpu_count())) as pool:
        result = pool.map(_run_foldx_repair, tqdm(pdb_fpaths, desc=f'\033[0;37;42m Repair for PDBs\033[0m'))
    success_count = sum(r[1] is None for r in result)
    print(f"{success_count} PDBs Repaired by FoldX")
    if pdb_fpaths != []:
        subprocess.run(f"rm {args.output_pdb_dir}/*.fxout", shell=True)

    # buildmodel
    mutant_file_wt, mutant_file_mt = create_individual_file(args.output_csv_path, args.output_individual_dir)
    _run_foldx_buildmodel_mt = partial(run_foldx_buildmodel_mt, pdb_dir=args.output_pdb_dir,
                                       foldx_dir=args.output_foldx_dir, output_wt_dir=args.output_wt_dir,
                                       output_mt_dir=args.output_mt_dir)
    with mp.Pool(min(MAX_CPU_cores, mp.cpu_count())) as pool:
        result = pool.map(_run_foldx_buildmodel_mt,
                          tqdm(mutant_file_mt, desc=f'\033[0;37;42m Buildmodel for mutant\033[0m'))
    success_count = sum(r[1] is None for r in result)
    print(f"{success_count} PDBs of mutant Buildmodel by FoldX")

    # AnalyseComplex
    df = pd.read_csv(args.output_csv_path, sep=',')
    df.replace("1.00E+96", "1E96", inplace=True)
    df.replace("1.00E+50", "1E50", inplace=True)
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

    # # AnalyseComplex for optimized
    # optimized_fpaths = [p for p in args.output_optimized_dir.glob('*.pdb') if not p.name.startswith('.')]
    # _run_foldx_analysecomplex_op = partial(run_foldx_analysecomplex_op, pdb_dir=args.output_optimized_dir,
    #                                        partner=complex_partner, output_dir=args.output_analysecomplex_op_dir)
    # with mp.Pool(min(MAX_CPU_cores, mp.cpu_count())) as pool:
    #     result = pool.map(_run_foldx_analysecomplex_op,
    #                       tqdm(optimized_fpaths, desc=f'\033[0;37;42m AnalyseComplex for optimized\033[0m'))
    # success_count = sum(r[1] is None for r in result)
    # print(f"{success_count} PDBs of optimized AnalyseComplex by FoldX")

    # update FoldX ddg
    df = pd.read_csv(args.output_csv_path, sep=',')
    df.replace("1.00E+96", "1E96", inplace=True)
    df.replace("1.00E+50", "1E50", inplace=True)
    df["ddG_FoldX"] = df.apply(lambda x: func1(x['#Pdb'], subset=subset), axis=1)  # Interaction
    logger.info(f'Writing to {prefix}/{subset}.csv')
    df.to_csv(f"{args.output_csv_path}", index=None)
    pearson = df[['ddG', 'ddG_FoldX']].corr('pearson').iloc[0, 1]
    spearman = df[['ddG', 'ddG_FoldX']].corr('spearman').iloc[0, 1]
    logger.info(f'[FoldX-ddg]: pearson={pearson:.4f}, spearman={spearman:.4f}')