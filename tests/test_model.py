import pandas as pd
from heart_disease.config import FEATURES
from heart_disease.model import build_candidates

def test_each_pipeline_fits_and_predicts():
    rows=[]
    for i in range(30):
        rows.append([40+i%20,i%2,1+i%4,110+i,180+i*2,i%2,i%3,130+i%40,i%2,float(i%5),1+i%3,float(i%4),float(3+(i%3)*2)])
    X=pd.DataFrame(rows,columns=FEATURES); y=pd.Series([i%2 for i in range(30)])
    for model in build_candidates().values():
        model.fit(X,y); assert model.predict_proba(X.iloc[:2]).shape==(2,2)
