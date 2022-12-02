from file_util import read_csv 
from CONST import RAW_DATASET_PATH, COL_NAMES

import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

def whiten(X):
    X = np.array(X)
    pca = PCA(whiten=True)
    whitened = pca.fit_transform(X)
    return whitened


def preprocess_dataset(file_path, col_names):
    raw_dataset = read_csv(file_path)
    X = []
    y = []
    
    for sample in raw_dataset:
        if len(sample) != 15:
            continue

        sample = [s.strip() for s in sample]  

        # y
        label = 1 if sample[-1] == "<=50K" else 0
        
        # age
        x_age = int(sample[0])

        # Create dummy variables for workclass
        x_workclass_private = 1 if sample[1] == "Private" else 0
        x_workclass_not_inc = 1 if sample[1] == "Self-emp-not-inc" else 0
        x_workclass_inc = 1 if sample[1] == "Self-emp-inc" else 0
        x_workclass_federal = 1 if sample[1] == "Federal-gov" else 0
        x_workclass_local = 1 if sample[1] == "Local-gov" else 0
        x_workclass_state = 1 if sample[1] == "State-gov" else 0
        x_workclass_wopay = 1 if sample[1] == "Without-pay" else 0
        x_workclass_nvwork = 1 if sample[1] == "Never-worked" else 0

        # fnlwgt
        x_fnlwgt = float(sample[2])
        
        # Create dummy variables for education
        x_edu_bachelor = 1 if sample[3] == "Bachelors" else 0
        x_edu_college = 1 if sample[3] == "Some-college" else 0
        x_edu_elevth = 1 if sample[3] == "11th" else 0
        x_edu_hsgrad = 1 if sample[3] == "HS-grad" else 0
        x_edu_prof = 1 if sample[3] == "Prof-school" else 0
        x_edu_acdm = 1 if sample[3] == "Assoc-acdm" else 0
        x_edu_voc = 1 if sample[3] == "Assoc-voc" else 0
        x_edu_ninth = 1 if sample[3] == "9th" else 0
        x_edu_seveneight = 1 if sample[3] == "7th-8th" else 0
        x_edu_twe = 1 if sample[3] == "12th" else 0
        x_edu_master = 1 if sample[3] == "Masters" else 0
        x_edu_firstfour = 1 if sample[3] == "1st-4t" else 0
        x_edu_ten = 1 if sample[3] == "10th" else 0
        x_edu_doc = 1 if sample[3] == "Doctorate" else 0 
        x_edu_fifsix = 1 if sample[3] == "5th-6th" else 0
        x_edu_presch = 1 if sample[3] == "Preschool" else 0
        
        # education num
        x_edu_num = int(sample[4])
        
        # Create dummy variables for marital-status
        x_marital_civ = 1 if sample[5] == "Married-civ-spouse" else 0
        x_marital_div = 1 if sample[5] == "Divorced" else 0
        x_martial_nev = 1 if sample[5] == "Never-married" else 0
        x_martial_sep = 1 if sample[5] == "Separated" else 0
        x_martial_wid = 1 if sample[5] == "Widowed" else 0
        x_martial_abs = 1 if sample[5] == "Married-spouse-absent" else 0
        x_martial_af = 1 if sample[5] == "Married-AF-spouse" else 0
        
        # Create dummpy variables for occupation
        x_occup_tech = 1 if sample[6] == "Tech-support" else 0 
        x_occup_craft = 1 if sample[6] == "Craft-repair" else 0
        x_occup_serv = 1 if sample[6] == "Other-service" else 0
        x_occup_sales = 1 if sample[6] == "Sales" else 0
        x_occup_exec = 1 if sample[6] == "Exec-managerial" else 0
        x_occup_prof = 1 if sample[6] == "Prof-specialty" else 0
        x_occup_handler = 1 if sample[6] == "Handlers-cleaners" else 0
        x_occup_mach = 1 if sample[6] == "Machine-op-inspct" else 0
        x_occup_adm = 1 if sample[6] == "Adm-clerical" else 0
        x_occup_farm = 1 if sample[6] == "Farming-fishing" else 0
        x_occup_trans = 1 if sample[6] == "Transport-moving" else 0
        x_occup_priv = 1 if sample[6] == "Priv-house-serv" else 0
        x_occup_prot = 1 if sample[6] == "Protective-serv" else 0
        x_occup_armed = 1 if sample[6] == "Armed-Forces" else 0
        
        # Create dummy variables for relationship
        x_relat_wif = 1 if sample[7] == "Wife" else 0 
        x_relat_child = 1 if sample[7] == "Own-child" else 0
        x_relat_hus = 1 if sample[7] == "Husband" else 0
        x_relat_notin = 1 if sample[7] == "Not-in-family" else 0
        x_relat_other = 1 if sample[7] == "Other-relative" else 0
        x_relat_unmar = 1 if sample[7] == "Unmarried" else 0
        
        # Create dummy variables for race
        x_race_white = 1 if sample[8] == "White" else 0
        x_race_asian = 1 if sample[8] == "Asian-Pac-Islander" else 0
        x_race_amer = 1 if sample[8] == "Amer-Indian-Eskimo" else 0
        x_race_other = 1 if sample[8] == "Other" else 0
        x_race_black = 1 if sample[8] == "Black" else 0
        
        # Create dummpy variables for sex
        x_sex_female = 1 if sample[9] == "Female" else 0
        x_sex_male = 1 if sample[9] == "Male" else 0
        
        # capital gain
        x_capital_gain = float(sample[10])
        
        # capital loss
        x_capital_loss = float(sample[11])
        
        # hours per week
        x_hours_week = float(sample[12])
        
        # Create dummpy variables for native country
        x_natv_us = 1 if sample[13] == "United-States" else 0
        x_natv_camb = 1 if sample[13] == "Cambodia" else 0
        x_natv_eng = 1 if sample[13] == "England" else 0
        x_natv_puerto = 1 if sample[13] == "Puerto-Rico" else 0
        x_natv_canada = 1 if sample[13] == "Canada" else 0
        x_natv_germ = 1 if sample[13] == "Germany" else 0
        x_natv_outus = 1 if sample[13] == "Outlying-US(Guam-USVI-etc)" else 0
        x_natv_india = 1 if sample[13] == "India" else 0
        x_natv_jap = 1 if sample[13] == "Japan" else 0
        x_natv_greece = 1 if sample[13] == "Greece" else 0
        x_natv_south = 1 if sample[13] == "South" else 0
        x_natv_china = 1 if sample[13] == "China" else 0
        x_natv_cuba = 1 if sample[13] == "Cuba" else 0
        x_natv_iran = 1 if sample[13] == "Iran" else 0
        x_natv_hond = 1 if sample[13] == "Honduras" else 0
        x_natv_phil = 1 if sample[13] == "Philippines" else 0
        x_natv_ital = 1 if sample[13] == "Italy" else 0
        x_natv_pol = 1 if sample[13] == "Poland" else 0
        x_natv_jam = 1 if sample[13] == "Jamaica" else 0
        x_natv_viet = 1 if sample[13] == "Vietnam" else 0
        x_natv_Mexico = 1 if sample[13] == "Mexico" else 0
        x_natv_port = 1 if sample[13] == "Portugal" else 0
        x_natv_irel = 1 if sample[13] == "Ireland" else 0
        x_natv_fran = 1 if sample[13] == "France" else 0
        x_natv_domi = 1 if sample[13] == "Dominican-Republic" else 0
        x_natv_laos = 1 if sample[13] == "Laos" else 0
        x_natv_ecuador = 1 if sample[13] == "Ecuador" else 0
        x_natv_taiwan = 1 if sample[13] == "Taiwan" else 0
        x_natv_Haiti = 1 if sample[13] == "Haiti" else 0
        x_natv_colum = 1 if sample[13] == "Columbia" else 0
        x_natv_hun = 1 if sample[13] == "Hungary" else 0
        x_natv_gua = 1 if sample[13] == "Guatemala" else 0
        x_natv_nica = 1 if sample[13] == "Nicaragua" else 0
        x_natv_scot = 1 if sample[13] == "Scotland" else 0
        x_natv_thai = 1 if sample[13] == "Thailand" else 0
        x_natv_yugo = 1 if sample[13] == "Yugoslavia" else 0
        x_natv_el_salv = 1 if sample[13] == "El-Salvador" else 0
        x_natv_tri = 1 if sample[13] == "Trinadad&Tobago" else 0
        x_natv_peru = 1 if sample[13] == "Peru" else 0
        x_natv_hong = 1 if sample[13] == "Hong" else 0 
        x_natv_Holand = 1 if sample[13] == "Holand-Netherlands" else 0
        
        X.append([x_age, x_workclass_private, x_workclass_not_inc, x_workclass_inc, \
            x_workclass_federal, x_workclass_local, x_workclass_state, x_workclass_wopay, \
            x_workclass_nvwork, x_fnlwgt, x_edu_bachelor, x_edu_college, x_edu_elevth, \
            x_edu_hsgrad, x_edu_prof, x_edu_acdm, x_edu_voc, x_edu_ninth, x_edu_seveneight, \
            x_edu_twe, x_edu_master, x_edu_firstfour, x_edu_ten, x_edu_doc, x_edu_fifsix, x_edu_presch, \
            x_edu_num, x_marital_civ, x_marital_div, x_martial_nev, x_martial_sep, x_martial_wid, x_martial_abs, \
            x_martial_af, x_occup_tech, x_occup_craft, x_occup_serv, x_occup_sales, x_occup_exec, x_occup_prof, \
            x_occup_handler, x_occup_mach, x_occup_adm, x_occup_farm, x_occup_trans, x_occup_priv, x_occup_prot, \
            x_occup_armed, x_relat_wif, x_relat_child, x_relat_hus, x_relat_notin, x_relat_other, x_relat_unmar, \
            x_race_white, x_race_asian, x_race_amer, x_race_other, x_race_black, x_sex_female, x_sex_male, \
            x_capital_gain, x_capital_loss, x_hours_week, x_natv_us, x_natv_camb, x_natv_eng, x_natv_puerto, \
            x_natv_canada, x_natv_germ, x_natv_outus, x_natv_india, x_natv_jap, x_natv_greece, x_natv_south, \
            x_natv_china, x_natv_cuba, x_natv_iran, x_natv_hond, x_natv_phil, x_natv_ital, x_natv_pol, x_natv_jam, \
            x_natv_viet, x_natv_Mexico, x_natv_port, x_natv_irel, x_natv_fran, x_natv_domi, x_natv_laos, \
            x_natv_ecuador, x_natv_taiwan, x_natv_Haiti, x_natv_colum, x_natv_hun, x_natv_gua, x_natv_nica, x_natv_scot, \
            x_natv_thai, x_natv_yugo, x_natv_el_salv, x_natv_tri, x_natv_peru, x_natv_hong, x_natv_Holand])
    
        y.append(label)

    # Conduct normalization on X
    X_normalized = whiten(X)

    # Shuffle X and y
    X_preproc, y = shuffle(X_normalized, y, random_state=0)

    return X_preproc, y
