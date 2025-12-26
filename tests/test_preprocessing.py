import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data_pipeline.preprocessing.preprocessing import preprocess_data

@patch('src.data_pipeline.preprocessing.preprocessing.np.save')
@patch('src.data_pipeline.preprocessing.preprocessing.pd.read_csv')
def test_preprocess_data(mock_read_csv, mock_save):
    # 1. Setup Mock Data
    mock_df = pd.DataFrame({
        'date': ['2016-07-01 00:00:00', '2016-07-01 01:00:00'],
        'OT': [10.0, 20.0],
        'HUFL': [1.0, 1.0], 'HULL': [1.0, 1.0], 
        'MUFL': [1.0, 1.0], 'MULL': [1.0, 1.0], 
        'LUFL': [1.0, 1.0], 'LULL': [1.0, 1.0]
    })
    mock_read_csv.return_value = mock_df

    # 2. Run Function
    # The path doesn't matter because we mock read_csv
    preprocess_data('dummy/path.csv')

    # 3. Verify Interactions
    mock_read_csv.assert_called_with('dummy/path.csv')
    mock_save.assert_called_once()
    
    # Verify logic (e.g. date conversion happened because index is now DatetimeIndex)
    # Note: We can't easily check the return value of preprocess_data in the current implementation 
    # because it returns None (implicit) ? Wait, the type hint says -> pd.DataFrame but there is no return statement!
