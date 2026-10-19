import pandas as pd
import pytest
from heart_disease.config import COLUMNS
from heart_disease.data import clean_data

def test_clean_data_converts_target_and_duplicates():
    row=[63,1,1,145,233,1,2,150,0,2.3,3,0,6,2]
    df=pd.DataFrame([row,row],columns=COLUMNS); cleaned=clean_data(df)
    assert len(cleaned)==1 and cleaned.loc[0,"target"]==1

def test_clean_data_rejects_missing_schema():
    with pytest.raises(ValueError,match="Missing required"):
        clean_data(pd.DataFrame({"age":[50]}))
