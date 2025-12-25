import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    对ETTh1数据集进行预处理，包括缺失值处理、特征缩放等。

    参数:
    file_path (str): ETTh1数据集的文件路径。

    返回:
    pd.DataFrame: 预处理后的数据集。
    """
    # 读取数据集
    df = pd.read_csv(file_path)
    # 处理缺失值
    df.fillna(method='ffill', inplace=True)
    # 特征缩放
    scaler = StandardScaler()
    df[['OT']] = scaler.fit_transform(df[['OT']])
    # 转换日期时间格式
    df['date'] = pd.to_datetime(df['date'])
    # 设置日期时间为索引
    df.set_index('date', inplace=True)
    # 重采样为1小时频率
    df = df.resample('1H').mean()
    # 处理缺失值（如果有）
    df.fillna(method='ffill', inplace=True)
    np.save('../../data/processed/ETTh1.npy', df.values)
    
    
    
    
    
    
    
    
    
    
