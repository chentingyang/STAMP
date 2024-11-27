
import pandas as pd

lookup = {
        "Normal": 0,
        "Attack": 1
    }

def remove_same_columns():
    input_file = r"G:\gitClone\papers\时序异常检测\gitOtherClone\ourModel\data\SWaT\SWaT_Dataset_Normal_v1.csv"
    normal = pd.read_csv(input_file)

    input_file = r"G:\gitClone\papers\时序异常检测\gitOtherClone\ourModel\data\SWaT\SWaT_Dataset_Attack_v0.csv"
    attack = pd.read_csv(input_file, sep=";")

    normal = normal.rename(columns={"Normal/Attack": "Attack"})
    attack = attack.rename(columns={"Normal/Attack": "Attack"})

    # Transform all columns into float64
    for i in list(normal):
        if i not in ['Timestamp', 'Attack']:
            normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
            normal[i] = normal[i].astype(float)
        if i == "Attack":
            normal[i] = normal[i].apply(lookup.get)

    for i in list(attack):
        if i not in ['Timestamp', 'Attack']:
            attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
            attack[i] = attack[i].astype(float)
        if i == "Attack":
            attack[i] = attack[i].apply(lookup.get)

    normal1 = normal.loc[:, (normal != normal.iloc[0]).any()]
    attack1 = attack.loc[:, (attack != attack.iloc[0]).any()]

    a = set(normal.columns) - set(normal1.columns)
    b = set(attack.columns) - set(attack1.columns)

    c = a.intersection(b)
    print("normal和attack列值重复的列的交集为:{0}".format(c))
    c = list(c)
    normal = normal.drop(c, axis=1)
    attack = attack.drop(c, axis=1)

    output_file_normal = r"G:\gitClone\papers\时序异常检测\gitOtherClone\ourModel\data\SWaT\SWaT_Dataset_normal.csv"
    output_file_attack = r"G:\gitClone\papers\时序异常检测\gitOtherClone\ourModel\data\SWaT\SWaT_Dataset_attack.csv"
    normal.to_csv(output_file_normal, index=False)
    attack.to_csv(output_file_attack, index=False)


def remove_same_columns_wadi():
    input_file = r"G:\gitClone\papers\时序异常检测\gitOtherClone\ourModel\data\WADI\normal.csv"
    normal = pd.read_csv(input_file)

    input_file = r"G:\gitClone\papers\时序异常检测\gitOtherClone\ourModel\data\WADI\attack.csv"
    attack = pd.read_csv(input_file)

    print("normal: ", normal.shape)
    print("attack: ", attack.shape)

    normal["label"] = [0. for _ in range(len(normal))]
    attack = attack.drop(["Date_Time"],axis=1)

    ### set columns
    cols = [x.split("\\")[-1] for x in normal.columns]
    normal.columns = cols
    attack.columns = cols

    normal1 = normal.loc[:, (normal != normal.iloc[0]).any()]
    attack1 = attack.loc[:, (attack != attack.iloc[0]).any()]

    a = set(normal.columns) - set(normal1.columns)
    b = set(attack.columns) - set(attack1.columns)

    c = a.intersection(b)
    print("normal和attack列值重复的列的交集为:{0}".format(c))
    c = list(c)
    normal = normal.drop(c, axis=1)
    attack = attack.drop(c, axis=1)

    print("normal: ", normal.shape)
    print("attack: ", attack.shape)
    print(normal.columns)

    output_file_normal = r"G:\gitClone\papers\时序异常检测\gitOtherClone\ourModel\data\WADI\normal_v1.csv"
    output_file_attack = r"G:\gitClone\papers\时序异常检测\gitOtherClone\ourModel\data\WADI\attack_v1.csv"
    # normal.to_csv(output_file_normal, index=False)
    # attack.to_csv(output_file_attack, index=False)

def preprocess4GDN(input_file, output_file, sep=None):
    # === Normal period ====
    if sep:
        normal = pd.read_csv(input_file, sep=sep)  # , nrows=1000)
    else:
        normal = pd.read_csv(input_file)

    normal = normal.rename(columns = {"Normal/Attack": "attack"})

    # print(list(normal))
    ## ['Timestamp', 'FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203', 'FIT201',
    # 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302',
    # 'MV303', 'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401', 'LIT401', 'P401', 'P402', 'P403', 'P404',
    # 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502',
    # 'PIT501', 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603', 'attack']

    # Transform all columns into float64
    for i in list(normal):
        if i not in ['Timestamp','attack']:
            normal[i] = normal[i].apply(lambda x: float(str(x).replace(",", ".")))
        if i == "attack":
            normal[i] = normal[i].apply(lookup.get)

    normal.to_csv(output_file, header=normal.columns)

if __name__ == '__main__':
    # remove_same_columns()
    remove_same_columns_wadi()

    ##['Timestamp', 'FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203',
    # 'FIT201', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'DPIT301', 'FIT301',
    # 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302', 'AIT401', 'AIT402', 'FIT401',
    # 'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504',
    # 'FIT501', 'FIT502', 'FIT503', 'FIT504', 'P501', 'P502', 'PIT501', 'PIT502', 'PIT503', 'FIT601',
    # 'P601', 'P602', 'P603', 'Normal/Attack']


    ## preprocess data for GDN

    # input_file = r"G:\gitClone\papers\时序异常检测\gitOtherClone\ourModel\data\SWaT\SWaT_Dataset_Normal_v1.csv"
    # output_file = r"G:\gitOtherClone\GDN-main\data\SWaT\train.csv"
    # preprocess4GDN(input_file,output_file,sep=None)
    #
    # input_file = r"G:\gitClone\papers\时序异常检测\gitOtherClone\ourModel\data\SWaT\SWaT_Dataset_Attack_v0.csv"
    # output_file = r"G:\gitOtherClone\GDN-main\data\SWaT\test.csv"
    # preprocess4GDN(input_file, output_file, sep=";")