import pandas as pd
from typing import List, Dict, Any

class DIANNParquetReader:
    """DIA-NN parquet 结果读取和解析类"""
    def __init__(self):
        self.df = None
    
    def read(self, lib_path: str, report_path: str) -> pd.DataFrame:
        """读取DIA-NN parquet结果文件并处理数据
        
        参数:
            lib_path: parquet文件路径
            report_path: parquet文件路径
            
        返回:
            处理后的DataFrame
        """

        def _read_df(file_path: str) -> pd.DataFrame:
            if file_path.endswith('.parquet'):
                return pd.read_parquet(file_path)
            elif file_path.endswith('.tsv'):
                return pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError("file_path must be a parquet or tsv file")
        
        lib_df = _read_df(lib_path)
        report_df = _read_df(report_path)

        # 只保留需要的列
        lib_needed_columns = ['Precursor.Id', 'Decoy', 'RT', 'IM', 'Precursor.Mz', 'Product.Mz']
        if not all(col in lib_df.columns for col in lib_needed_columns):
            raise ValueError(f"lib_path must contain the following columns: {lib_needed_columns}")
        lib_df = lib_df[lib_needed_columns]
        lib_df = lib_df.rename(columns={
            'RT': 'iRT',
            'IM': 'iIM'
        })

        # 只保留需要的列
        report_needed_columns = ['Precursor.Id', 'RT', 'IM', 'RT.Start', 'RT.Stop']
        if not all(col in report_df.columns for col in report_needed_columns):
            raise ValueError(f"report_path must contain the following columns: {report_needed_columns}")
        report_df = report_df[report_needed_columns]

        # 合并lib和rt_report文件
        lib_df = pd.merge(lib_df, report_df, on='Precursor.Id', how='left')

        # 只保留需要的列
        needed_columns = ['Precursor.Id', 'Decoy', 'RT', 'iRT', 'iIM', 'IM', 'RT.Start', 'RT.Stop', 'Precursor.Mz', 'Product.Mz']
        if not all(col in lib_df.columns for col in needed_columns):
            raise ValueError(f"lib_path must contain the following columns: {needed_columns}")
        lib_df = lib_df[needed_columns].copy()

        lib_df = lib_df.rename(columns={
            'Precursor.Id': 'pr_id', 
            'Decoy': 'decoy', 
            'RT': 'rt', 
            'iRT': 'irt',
            'iIM': 'iIM',
            'IM': 'IM',
            'RT.Start': 'rt_start',
            'RT.Stop': 'rt_stop',
            'Precursor.Mz': 'precursor_mz', 
            'Product.Mz': 'fragment_mz'
        })

        lib_df = lib_df.groupby(['pr_id', 'decoy', 'rt', 'irt', 'iIM', 'IM', 'rt_start', 'rt_stop', 'precursor_mz'], dropna=False)['fragment_mz'].agg(lambda x: ','.join(map(str, x))).reset_index()

        # 解析肽段ID，提取序列和电荷
        peptide_info = [self._parse_peptide_id(pid) for pid in lib_df['pr_id']]
        lib_df['peptide'] = [info[0] for info in peptide_info]
        lib_df['charge'] = [info[1] for info in peptide_info]

        #转化标签
        lib_df['label'] = lib_df['decoy'].apply(lambda x: 1 if x == 0 else 0)   

        self.df = lib_df[['label', 'peptide', 'charge', 'precursor_mz', 'rt', 'irt', 'iIM', 'IM', 'rt_start', 'rt_stop', 'fragment_mz']]

        return self.df
    
    def get_all_peptide_info(self) -> pd.DataFrame:
        """获取所有肽段信息"""
        if self.df is None:
            raise ValueError("请先调用read方法读取数据")
        return self.df

    def _parse_peptide_id(self, peptide_id: str) -> tuple:
        """解析肽段ID，提取序列和电荷"""
        # 假设格式为 "序列+电荷"，如 "AAAAAAAAAAAAAAAGAGAGAK3"
        charge_str = ''
        for char in peptide_id[::-1]:
            if char.isdigit():
                charge_str += char
            else:
                break
        charge = int(charge_str[::-1])
        sequence = peptide_id[:-len(charge_str)]
        return sequence, charge