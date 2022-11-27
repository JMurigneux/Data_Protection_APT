import numpy as np

def raw_expand(df):
    rows_nb,_=df.shape

    for row in range(rows_nb):
        spl=df["raw"][row].split(' ') #on obtient la liste des col_name=value
        spl2=[elem.split("=") for elem in spl if len(elem.split("="))>1] # chaque element est une liste de forme [col_name, value]

        for cpl in spl2:
            if cpl[0] not in df.columns: #si la colonne n'a jamais été vue on la rajoute remplie de NaN
                df[cpl[0]]=np.nan
            df[cpl[0]][row]=cpl[1]