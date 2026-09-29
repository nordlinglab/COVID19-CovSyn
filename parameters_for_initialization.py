import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pathlib import Path
from scipy.optimize import lsq_linear
from rw_data_processing import *
from Data_synthesize import *
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

plt.style.use("./rw_visualization.mplstyle")

def get_household_data(demographic_data_path):
    family_size_data = pd.read_excel(demographic_data_path/'縣市戶數結構.xls', sheet_name='110')
    family_size_data.columns = family_size_data.iloc[1]
    family_size_data = family_size_data[3:23+1]
    family_size_data = family_size_data[family_size_data['區  域  別'] != '臺  灣  省']
    family_size_data = family_size_data[family_size_data['區  域  別'] != '福  建  省']
    family_size_data = family_size_data[family_size_data['區  域  別'] != '澎  湖  縣']

    # Load total number of famaliy each city
    family_number_data = pd.read_excel(demographic_data_path/'縣市村里鄰戶數及人口數-110年.xls')
    family_number_data.columns = family_number_data.iloc[1]
    family_number_data = family_number_data[4:28]
    family_number_data = family_number_data[family_number_data['區域別'] != '臺灣省']
    family_number_data = family_number_data[family_number_data['區域別'] != '福建省']
    family_number_data = family_number_data[family_number_data['區域別'] != '澎湖縣']
    family_number_data = family_number_data[family_number_data['區域別'] != '連江縣']
    family_number_data = family_number_data[family_number_data['區域別'] != '金門縣']
    family_number = family_number_data['人口數'].to_numpy()

    r2_list = []
    rmse_list = []
    family_size_dict = {}
    for i, city in enumerate(family_size_data['區  域  別']):
        family_size_array = family_size_data[family_size_data['區  域  別'] == city].to_numpy()[0][2:-1]
        
        # Get observed values
        people_higher_than_6 = family_number[i] - np.sum(family_size_array[0:-1]*np.arange(1, 6))
        y_obs = np.array([people_higher_than_6, family_size_array[-1]])
        
        # Get predicted values 
        A = np.array([[j, 1] for j in range(6, 13)])
        res = lsq_linear(A.T, y_obs, bounds=(0, 10**10), lsmr_tol='auto', verbose=0)
        y_pred = A.T @ res.x
        
        # Calculate R^2 and RMSE
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        ss_res = np.sum((y_obs - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r2_list.append(r2)
        rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
        rmse_list.append(rmse)
        
        family_size_array_tmp = np.append(family_size_array[0:-1], np.array(res.x))
        family_size_array = family_size_array_tmp
        family_size_dict[city.replace(' ', '')] = np.array(family_size_array/sum(family_size_array), dtype=float)
    print(f'Max family size RMSE: {max(rmse_list):.2e}')

    return(family_size_dict)

def get_school_data(demographic_data_path):
    school_p = {}

    # Elementary school
    elementary_school_data = pd.read_excel(demographic_data_path/'國民小學校別資料.xls')
    elementary_school_data.columns = elementary_school_data.iloc[1]
    elementary_school_data = elementary_school_data[2::]
    elementary_school_data = elementary_school_data[elementary_school_data['縣市名稱'] != '金門縣']
    elementary_school_data = elementary_school_data[elementary_school_data['縣市名稱'] != '連江縣']
    elementary_school_data = elementary_school_data[elementary_school_data['縣市名稱'] != '澎湖縣']

    class_index = elementary_school_data.columns[6:11+1]
    class_index_english = ['1st-grade', '2nd-grade',
                        '3rd-grade', '4th-grade', '5th-grade', '6th-grade']
    student_number_index = elementary_school_data.columns[15:26+1]
    for i in range(len(class_index)):
        plt.figure(figsize=(12, 4))
        # Remove empty schools
        elementary_school_data_temp = elementary_school_data[elementary_school_data[class_index[i]] != 0]
        student_per_class = (elementary_school_data_temp[student_number_index[2*i]] +
                            elementary_school_data_temp[student_number_index[2*i+1]])/elementary_school_data_temp[class_index[i]]
        n = plt.hist(student_per_class, bins=range(int(max(student_per_class))))
        probability = np.zeros(int(n[1].max()) + 1)
        probability[n[1][0:-1].astype(int)] = n[0]/sum(n[0])
        school_p[i+7] = probability

    # Junior high school
    junior_high_school_data = pd.read_excel(demographic_data_path/'國民中學校別資料.xlsx')
    junior_high_school_data.columns = junior_high_school_data.iloc[1]
    junior_high_school_data = junior_high_school_data[2::]
    junior_high_school_data = junior_high_school_data[junior_high_school_data['縣市名稱'] != '金門縣']
    junior_high_school_data = junior_high_school_data[junior_high_school_data['縣市名稱'] != '連江縣']
    junior_high_school_data = junior_high_school_data[junior_high_school_data['縣市名稱'] != '澎湖縣']

    class_index = junior_high_school_data.columns[6:8+1]
    class_index_english = ['7th-grade', '8th-grade', '9th-grade']
    student_number_index = junior_high_school_data.columns[12:17+1]
    for i in range(len(class_index)):
        plt.figure(figsize=(12, 4))
        # Remove empty schools
        junior_high_school_data_temp = junior_high_school_data[
            junior_high_school_data[class_index[i]] != 0]
        student_per_class = (junior_high_school_data_temp[student_number_index[2*i]] +
                            junior_high_school_data_temp[student_number_index[2*i+1]])/junior_high_school_data_temp[class_index[i]]
        n = plt.hist(student_per_class, bins=range(int(max(student_per_class))))
        probability = np.zeros(int(n[1].max())+1)
        probability[n[1][0:-1].astype(int)] = n[0]/sum(n[0])
        school_p[i+13] = probability

        plt.xticks(np.arange(0, int(max(student_per_class)), 5))
        plt.xlabel('Amount of ' + class_index_english[i] + ' student per class')
        plt.ylabel('Amount of school')
        plt.close()

    # Senior high school
    senior_high_school_data = pd.read_excel(demographic_data_path/'高級中等學校校別資料檔.xls')
    senior_high_school_data.columns = senior_high_school_data.iloc[1]
    senior_high_school_data = senior_high_school_data[2::]
    senior_high_school_data = senior_high_school_data[senior_high_school_data['縣市名稱'] != '金門縣']
    senior_high_school_data = senior_high_school_data[senior_high_school_data['縣市名稱'] != '連江縣']
    senior_high_school_data = senior_high_school_data[senior_high_school_data['縣市名稱'] != '澎湖縣']
    # Remove 附設國中部
    senior_high_school_data = senior_high_school_data[~(
        senior_high_school_data['學程(等級)別'] == 'J')]
    
    class_index = senior_high_school_data.columns[9:11+1]
    class_index_english = ['1st-grade', '2nd-grade', '3rd-grade']
    student_number_index = senior_high_school_data.columns[16:21+1]
    for i in range(len(class_index)):
        plt.figure(figsize=(12, 4))
        # Remove empty schools
        senior_high_school_data_temp = senior_high_school_data[
            senior_high_school_data[class_index[i]] != 0]
        student_per_class = (senior_high_school_data_temp[student_number_index[2*i]] +
                            senior_high_school_data_temp[student_number_index[2*i+1]])/senior_high_school_data_temp[class_index[i]]
        n = plt.hist(student_per_class, bins=range(int(max(student_per_class))))
        probability = np.zeros(int(n[1].max())+1)
        probability[n[1][0:-1].astype(int)] = n[0]/sum(n[0])
        school_p[i+16] = probability

    # University
    university_data = pd.read_excel(demographic_data_path/'大專校院各校科系別學生數.xlsx')
    university_data.columns = university_data.iloc[1]
    university_data = university_data[2::]
    # Only extract bachelor students
    university_data = university_data[university_data['等級別'] == 'B 學士']
    university_data = university_data[university_data['縣市名稱'] != '71 金門縣']

    # class_index = university_data.columns[9:22+1]
    class_index_english = ['1st-grade', '2nd-grade', '3rd-grade', '4th-grade']
    student_number_index = university_data.columns[9:16+1]
    for i in range(len(class_index_english)):
        plt.figure(figsize=(12, 4))
        student_number = university_data[student_number_index[2*i]
                                        ] + university_data[student_number_index[2*i+1]]
        student_number = student_number[student_number > 0]  # Remove empty grade
        n = plt.hist(student_number, bins=range(int(max(student_number))))
        probability = np.zeros(int(n[1].max())+1)
        probability[n[1][0:-1].astype(int)] = n[0]/sum(n[0])
        school_p[i+19] = probability

    return(school_p, elementary_school_data, junior_high_school_data, senior_high_school_data, university_data)

def get_workplace_data(demographic_data_path):
    workplace_p = {}

    workplace_data = pd.read_excel(demographic_data_path/'工業及服務業企業單位經營概況.xls', sheet_name='18')
    workplace_data = workplace_data.iloc[15:-1, 1:17+1]
    workplace_data = workplace_data.drop(['Unnamed: 11'], axis=1)

    headers = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49',
           '50~99', '100~199', '200~299', '300~499', '500~999', '>1000']
    for row in workplace_data.iloc:
        if type(row.iloc[0]) == str:
            number_of_companys = row.iloc[3:14+1]
            probability = number_of_companys.to_numpy()/sum(number_of_companys.to_numpy())
            workplace_p_tmp = np.zeros(1000)
            # 1~999
            workplace_p_tmp[1:5] = probability[0]/4
            workplace_p_tmp[5:10] = probability[1]/5
            workplace_p_tmp[10:20] = probability[2]/10
            workplace_p_tmp[20:30] = probability[3]/10
            workplace_p_tmp[30:40] = probability[4]/10
            workplace_p_tmp[40:50] = probability[5]/10
            workplace_p_tmp[50:100] = probability[6]/50
            workplace_p_tmp[100:200] = probability[7]/100
            workplace_p_tmp[200:300] = probability[8]/100
            workplace_p_tmp[300:500] = probability[9]/200
            workplace_p_tmp[500:1000] = probability[10]/500
            # >1000
            if probability[11] != 0:
                append_numbers = np.ceil(probability[11]/workplace_p_tmp[-1])
                workplace_p_tmp = np.append(workplace_p_tmp, np.ones(
                    int(append_numbers))*(probability[11]/append_numbers))
            workplace_p_tmp = np.array(workplace_p_tmp, dtype=float)

            workplace_p[row.iloc[0]] = workplace_p_tmp

    workplace_data = pd.read_excel(demographic_data_path/'工業及服務業企業單位經營概況.xls', sheet_name='18-2')
    workplace_data = workplace_data.iloc[14:-1, 1:17+1]
    workplace_data = workplace_data.drop(['Unnamed: 11'], axis=1)

    headers = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49',
           '50~99', '100~199', '200~299', '300~499', '500~999', '>1000']
    for row in workplace_data.iloc:
        if type(row.iloc[0]) == str:
            number_of_companys = row.iloc[3:14+1]
            probability = number_of_companys.to_numpy()/sum(number_of_companys.to_numpy())
            workplace_p_tmp = np.zeros(1000)
            # 1~999
            workplace_p_tmp[1:5] = probability[0]/4
            workplace_p_tmp[5:10] = probability[1]/5
            workplace_p_tmp[10:20] = probability[2]/10
            workplace_p_tmp[20:30] = probability[3]/10
            workplace_p_tmp[30:40] = probability[4]/10
            workplace_p_tmp[40:50] = probability[5]/10
            workplace_p_tmp[50:100] = probability[6]/50
            workplace_p_tmp[100:200] = probability[7]/100
            workplace_p_tmp[200:300] = probability[8]/100
            workplace_p_tmp[300:500] = probability[9]/200
            workplace_p_tmp[500:1000] = probability[10]/500
            # >1000
            if probability[11] != 0:
                append_numbers = np.ceil(probability[11]/workplace_p_tmp[-1])
                workplace_p_tmp = np.append(workplace_p_tmp, np.ones(
                    int(append_numbers))*(probability[11]/append_numbers))
            workplace_p_tmp = np.array(workplace_p_tmp, dtype=float)

            workplace_p[row.iloc[0]] = workplace_p_tmp

    workplace_data = pd.read_excel(demographic_data_path/'工業及服務業企業單位經營概況.xls', sheet_name='18-3')
    workplace_data = workplace_data.iloc[14:-1, 1:17+1]
    workplace_data = workplace_data.drop(['Unnamed: 11'], axis=1)

    headers = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49',
           '50~99', '100~199', '200~299', '300~499', '500~999', '>1000']
    for row in workplace_data.iloc:
        if type(row.iloc[0]) == str:
            number_of_companys = row.iloc[3:14+1]
            probability = number_of_companys.to_numpy()/sum(number_of_companys.to_numpy())
            workplace_p_tmp = np.zeros(1000)
            # 1~999
            workplace_p_tmp[1:5] = probability[0]/4
            workplace_p_tmp[5:10] = probability[1]/5
            workplace_p_tmp[10:20] = probability[2]/10
            workplace_p_tmp[20:30] = probability[3]/10
            workplace_p_tmp[30:40] = probability[4]/10
            workplace_p_tmp[40:50] = probability[5]/10
            workplace_p_tmp[50:100] = probability[6]/50
            workplace_p_tmp[100:200] = probability[7]/100
            workplace_p_tmp[200:300] = probability[8]/100
            workplace_p_tmp[300:500] = probability[9]/200
            workplace_p_tmp[500:1000] = probability[10]/500
            # >1000
            if probability[11] != 0:
                append_numbers = np.ceil(probability[11]/workplace_p_tmp[-1])
                workplace_p_tmp = np.append(workplace_p_tmp, np.ones(
                    int(append_numbers))*(probability[11]/append_numbers))
            workplace_p_tmp = np.array(workplace_p_tmp, dtype=float)

            workplace_p[row.iloc[0]] = workplace_p_tmp

    workplace_data = pd.read_excel(demographic_data_path/'工業及服務業企業單位經營概況.xls', sheet_name='18-4')
    workplace_data = workplace_data.iloc[14:-1, 1:17+1]
    workplace_data = workplace_data.drop(['Unnamed: 11'], axis=1)

    headers = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49',
           '50~99', '100~199', '200~299', '300~499', '500~999', '>1000']
    for row in workplace_data.iloc:
        if type(row.iloc[0]) == str:
            number_of_companys = row.iloc[3:14+1]
            probability = number_of_companys.to_numpy()/sum(number_of_companys.to_numpy())
            workplace_p_tmp = np.zeros(1000)
            # 1~999
            workplace_p_tmp[1:5] = probability[0]/4
            workplace_p_tmp[5:10] = probability[1]/5
            workplace_p_tmp[10:20] = probability[2]/10
            workplace_p_tmp[20:30] = probability[3]/10
            workplace_p_tmp[30:40] = probability[4]/10
            workplace_p_tmp[40:50] = probability[5]/10
            workplace_p_tmp[50:100] = probability[6]/50
            workplace_p_tmp[100:200] = probability[7]/100
            workplace_p_tmp[200:300] = probability[8]/100
            workplace_p_tmp[300:500] = probability[9]/200
            workplace_p_tmp[500:1000] = probability[10]/500
            # >1000
            if probability[11] != 0:
                append_numbers = np.ceil(probability[11]/workplace_p_tmp[-1])
                workplace_p_tmp = np.append(workplace_p_tmp, np.ones(
                    int(append_numbers))*(probability[11]/append_numbers))
            workplace_p_tmp = np.array(workplace_p_tmp, dtype=float)

            workplace_p[row.iloc[0]] = workplace_p_tmp

    workplace_data = pd.read_excel(demographic_data_path/'工業及服務業企業單位經營概況.xls', sheet_name='18-5')
    workplace_data = workplace_data.iloc[14:-1, 1:17+1]
    workplace_data = workplace_data.drop(['Unnamed: 11'], axis=1)

    headers = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49',
           '50~99', '100~199', '200~299', '300~499', '500~999', '>1000']
    for row in workplace_data.iloc:
        if type(row.iloc[0]) == str:
            number_of_companys = row.iloc[3:14+1]
            probability = number_of_companys.to_numpy()/sum(number_of_companys.to_numpy())
            workplace_p_tmp = np.zeros(1000)
            # 1~999
            workplace_p_tmp[1:5] = probability[0]/4
            workplace_p_tmp[5:10] = probability[1]/5
            workplace_p_tmp[10:20] = probability[2]/10
            workplace_p_tmp[20:30] = probability[3]/10
            workplace_p_tmp[30:40] = probability[4]/10
            workplace_p_tmp[40:50] = probability[5]/10
            workplace_p_tmp[50:100] = probability[6]/50
            workplace_p_tmp[100:200] = probability[7]/100
            workplace_p_tmp[200:300] = probability[8]/100
            workplace_p_tmp[300:500] = probability[9]/200
            workplace_p_tmp[500:1000] = probability[10]/500
            # >1000
            if probability[11] != 0:
                append_numbers = np.ceil(probability[11]/workplace_p_tmp[-1])
                workplace_p_tmp = np.append(workplace_p_tmp, np.ones(
                    int(append_numbers))*(probability[11]/append_numbers))
            workplace_p_tmp = np.array(workplace_p_tmp, dtype=float)

            workplace_p[row.iloc[0]] = workplace_p_tmp

    workplace_data = pd.read_excel(demographic_data_path/'工業及服務業企業單位經營概況.xls', sheet_name='18-6')
    workplace_data = workplace_data.iloc[14:-1, 1:17+1]
    workplace_data = workplace_data.drop(['Unnamed: 11'], axis=1)

    headers = ['<5', '5~9', '10~19', '20~29', '30~39', '40~49',
           '50~99', '100~199', '200~299', '300~499', '500~999', '>1000']
    for row in workplace_data.iloc:
        if type(row.iloc[0]) == str:
            number_of_companys = row.iloc[3:14+1]
            probability = number_of_companys.to_numpy()/sum(number_of_companys.to_numpy())
            workplace_p_tmp = np.zeros(1000)
            # 1~999
            workplace_p_tmp[1:5] = probability[0]/4
            workplace_p_tmp[5:10] = probability[1]/5
            workplace_p_tmp[10:20] = probability[2]/10
            workplace_p_tmp[20:30] = probability[3]/10
            workplace_p_tmp[30:40] = probability[4]/10
            workplace_p_tmp[40:50] = probability[5]/10
            workplace_p_tmp[50:100] = probability[6]/50
            workplace_p_tmp[100:200] = probability[7]/100
            workplace_p_tmp[200:300] = probability[8]/100
            workplace_p_tmp[300:500] = probability[9]/200
            workplace_p_tmp[500:1000] = probability[10]/500
            # >1000
            if probability[11] != 0:
                append_numbers = np.ceil(probability[11]/workplace_p_tmp[-1])
                workplace_p_tmp = np.append(workplace_p_tmp, np.ones(
                    int(append_numbers))*(probability[11]/append_numbers))
            workplace_p_tmp = np.array(workplace_p_tmp, dtype=float)

            workplace_p[row.iloc[0]] = workplace_p_tmp

    return(workplace_p)

def get_health_care_data(demographic_data_path):
    health_care_data = pd.read_excel(demographic_data_path/'醫院平均每日醫療服務量統計.xls')
    health_care_data.columns = health_care_data.iloc[3]
    health_care_data = health_care_data[9::]
    health_care_data = health_care_data.drop(['合計'], axis=1)

    # Clean colunms and rows
    health_care_data = health_care_data.drop(16)  # Drop 非公立醫院
    health_care_data = health_care_data.drop(23)  # Drop 私立牙醫醫院
    hospital_lists = health_care_data.iloc[:, 0].to_list()  # Save hospital names
    hospital_numbers = health_care_data.iloc[:, 1].to_numpy()
    # Clean data
    for i in range(len(hospital_lists)):
        hospital_lists[i] = hospital_lists[i][2::]
    health_care_data = health_care_data.iloc[:, 2:-1]

    # Hospital name probability
    hospital_p = health_care_data.sum(axis=1).to_numpy()
    hospital_p = np.array(hospital_p/sum(hospital_p), dtype=float)
    hospital_p = [hospital_lists, hospital_p]

    hospital_size_p = {}
    hospital_sizes = {}
    for i, hospital_list in enumerate(hospital_lists):
        if sum(health_care_data.iloc[i].to_numpy()) > 0:
            hospital_size_p[hospital_list] = np.array(health_care_data.iloc[i].to_numpy(
            )/sum(health_care_data.iloc[i].to_numpy()), dtype=float)
        else:
            hospital_size_p[hospital_list] = 0
        hospital_sizes[hospital_list] = np.array(
            health_care_data.iloc[i].to_numpy()/hospital_numbers[i], dtype=float)
        
    return(hospital_p, hospital_size_p, hospital_sizes)

def get_age_gender_data(demographic_data_path):
    age_gender_data = pd.read_csv(demographic_data_path/'Single_Age_Population_2022-04-25.csv')
    age_male_data = age_gender_data[age_gender_data['項目']
                                    == '男性']['2022'].to_numpy()
    age_female_data = age_gender_data[age_gender_data['項目']
                                    == '女性']['2022'].to_numpy()
    # Age
    age_p = np.array(age_male_data) + np.array(age_female_data)
    age_p = age_p/np.sum(age_p)

    # Gender
    gender_age_groups = np.vstack([age_male_data, age_female_data]).T
    gender_p = np.empty([0, 2])
    for i in gender_age_groups:
        gender_p = np.vstack([gender_p, i/sum(i)])

    return(age_p, gender_p, age_male_data, age_female_data)


def get_student_rate(elementary_school_data, junior_high_school_data, senior_high_school_data, university_data, age_male_data, age_female_data):
    # Decide the probability of the person is a student or not.
    male_student_p = np.zeros(101)
    female_student_p = np.zeros(101)

    # Age 7 to 12 (elementary school)
    male_elementary_student_p = elementary_school_data[[
        '1年級男', '2年級男', '3年級男', '4年級男', '5年級男', '6年級男']].sum().to_numpy()/age_male_data[7:12+1]
    female_elementary_student_p = elementary_school_data[[
        '1年級女', '2年級女', '3年級女', '4年級女', '5年級女', '6年級女']].sum().to_numpy()/age_female_data[7:12+1]


    male_student_p[7:12+1] = male_elementary_student_p
    female_student_p[7:12+1] = female_elementary_student_p

    # Age 13 to 15 (junior high school)
    male_junior_high_school_student_p = junior_high_school_data[[
        '7年級男', '8年級男', '9年級男']].sum().to_numpy()/age_male_data[13:15+1]
    female_junior_high_school_student_p = junior_high_school_data[[
        '7年級女', '8年級女', '9年級女']].sum().to_numpy()/age_female_data[13:15+1]

    male_student_p[13:15+1] = male_junior_high_school_student_p
    female_student_p[13:15+1] = female_junior_high_school_student_p

    # Age 16 to 18 (senior high school)
    male_senior_high_school_p = senior_high_school_data[[
        '一年級男', '二年級男', '三年級男']].sum().to_numpy()/age_male_data[16:18+1]
    female_senior_high_school_p = senior_high_school_data[[
        '一年級女', '二年級女', '三年級女']].sum().to_numpy()/age_female_data[16:18+1]

    male_student_p[16:18+1] = male_senior_high_school_p
    female_student_p[16:18+1] = female_senior_high_school_p

    # Age 19 to 22 (university)
    male_university_p = university_data[[
        '一年級男生', '二年級男生', '三年級男生', '四年級男生']].sum().to_numpy()/age_male_data[19:22+1]
    female_university_p = university_data[[
        '一年級女生', '二年級女生', '三年級女生', '四年級女生']].sum().to_numpy()/age_female_data[19:22+1]

    male_student_p[19:22+1] = male_university_p
    female_student_p[19:22+1] = female_university_p

    # Replace >1 to 1
    male_student_p[male_student_p > 1] = 1
    female_student_p[female_student_p > 1] = 1

    student_p = np.vstack([male_student_p, female_student_p])

    return(student_p)

def get_employment_rate(demographic_data_path):
    # Emplolyment rate in each age group
    employment_data = pd.read_excel(demographic_data_path/'就業率-年齡別-20220428.xlsx')
    employment_data.columns = employment_data.iloc[2]
    employment_data = employment_data.iloc[3::]

    employment_p = employment_data.to_numpy()
    employment_p = employment_p[0][1::]

    male_employment_p = employment_p[0:9+1]/100
    female_employment_p = employment_p[10::]/100

    # Repeat emelent in the same age group
    male_employment_p = np.repeat(male_employment_p, 5)  # Split age group
    female_employment_p = np.repeat(female_employment_p, 5)  # Split age group

    # Zero padding
    male_employment_p = np.insert(
        male_employment_p, obj=0, values=np.zeros(15))  # Zero pading for age under 15
    male_employment_p = np.append(male_employment_p, np.zeros(
        36))  # Append last value for age above 65
    female_employment_p = np.insert(
        female_employment_p, obj=0, values=np.zeros(15))  # Zero pading for age under 15
    female_employment_p = np.append(female_employment_p, np.zeros(
        36))  # Append last value for age above 65

    employment_p = np.vstack([male_employment_p, female_employment_p])

    # Part-time and full-time job probability
    job_type_data = pd.read_excel(demographic_data_path/'mtable19.xlsx')
    job_type_data = job_type_data.iloc[9:29]

    # Drop 工業 and 服務業
    job_type_data = job_type_data[~(
        job_type_data['Unnamed: 1'] == '工業\n  Goods-Producing Industries')]
    job_type_data = job_type_data[~(
        job_type_data['Unnamed: 1'] == '服務業\n  Services-Producing Industries')]
    
    male_job_type_data = job_type_data.iloc[:, 13:14+1]
    female_job_type_data = job_type_data.iloc[:, 23:24+1]

    job_list = job_type_data['Unnamed: 1'].to_numpy()

    male_part_time_job_p = male_job_type_data['Unnamed: 14'].replace(
        '-', 0).to_numpy(dtype=float)/sum(male_job_type_data['Unnamed: 14'].replace('-', 0).to_numpy(dtype=float))
    male_full_time_job_p = male_job_type_data['Unnamed: 13'].replace(
        '-', 0).to_numpy(dtype=float)/sum(male_job_type_data['Unnamed: 13'].replace('-', 0).to_numpy(dtype=float))

    female_part_time_job_p = female_job_type_data['Unnamed: 24'].replace(
        '-', 0).to_numpy(dtype=float)/sum(female_job_type_data['Unnamed: 24'].replace('-', 0).to_numpy(dtype=float))
    female_full_time_job_p = female_job_type_data['Unnamed: 23'].replace(
        '-', 0).to_numpy(dtype=float)/sum(female_job_type_data['Unnamed: 23'].replace('-', 0).to_numpy(dtype=float))

    part_time_job_p = np.vstack([male_part_time_job_p, female_part_time_job_p])
    full_time_job_p = np.vstack([male_full_time_job_p, female_full_time_job_p])

    job_p = {}

    # Clean job_list
    for i in range(len(job_list)):
        job_list[i] = job_list[i].split('\n')[0].replace(' ', '')
    job_list = job_list[job_list != '農、林、漁、牧業']
    job_list = job_list[job_list != '公共行政及國防；強制性社會安全']
    job_list[8] = '出版、影音製作、傳播及資通訊服務業'
    job_list[9] = '金融及保險業、強制性社會安全'
    job_list[13] = '教育業(註)'


    job_p['job_list'] = job_list

    part_time_job_p = np.delete(
        part_time_job_p, (0, 14), axis=1)  # Remove non exist job
    row_sums = part_time_job_p.sum(axis=1)
    part_time_job_p = part_time_job_p/row_sums[:, np.newaxis]
    job_p['part_time_job_p'] = part_time_job_p

    full_time_job_p = np.delete(
        full_time_job_p, (0, 14), axis=1)  # Remove non exist job
    row_sums = full_time_job_p.sum(axis=1)
    full_time_job_p = full_time_job_p/row_sums[:, np.newaxis]
    job_p['full_time_job_p'] = full_time_job_p


    return(employment_p, job_p)

    


def get_municipality_data(demographic_data_path):
    municipality_data = pd.read_excel(demographic_data_path/'municipality_data.xls', sheet_name='02-縣市別')
    municipality_data.columns = municipality_data.iloc[1]
    municipality_data = municipality_data[4:24+1]
    municipality_data = municipality_data[municipality_data['區  域  別'] != '臺灣省']
    municipality_data = municipality_data[municipality_data['區  域  別'] != '福建省']
    municipality_data = municipality_data[municipality_data['區  域  別'] != '澎湖縣']
    municipality_data = municipality_data[municipality_data['區  域  別'] != '金門縣']
    municipality_data = municipality_data[municipality_data['區  域  別'] != '連江縣']
    municipality_data = municipality_data[municipality_data['區  域  別'] != '東沙群島']
    municipality_data = municipality_data[municipality_data['區  域  別'] != '南沙群島']

    municipality_data = dict(
        zip(municipality_data['區  域  別'], municipality_data['人　　口　　數']))

    return(municipality_data)

def get_demographic_data(demographic_data_path, save_path):
    print('Loading family size data')
    family_size_dict = get_household_data(demographic_data_path)
    print('Loading school data')
    school_p, elementary_school_data, junior_high_school_data, senior_high_school_data, university_data = get_school_data(demographic_data_path)
    print('Loading workplace data')
    workplace_p = get_workplace_data(demographic_data_path)
    print('Loading health care data')
    hospital_p, hospital_size_p, hospital_sizes = get_health_care_data(demographic_data_path)
    print('Loading municipality data')
    municipality_data = get_municipality_data(demographic_data_path)
    print('Loading age, student, and employment data')
    age_p, gender_p, age_male_data, age_female_data = get_age_gender_data(demographic_data_path)
    student_p = get_student_rate(elementary_school_data, junior_high_school_data, senior_high_school_data, university_data, age_male_data, age_female_data)
    employment_p, job_p = get_employment_rate(demographic_data_path)
    
    # NOTE: the census distributions are saved RAW. The person-weighting of household,
    # class and enterprise sizes (B29, B30, B31) is applied at load time by
    # apply_person_weighting() in Data_synthesis_main.py, so that this file keeps the
    # original published distributions and the weighting stays visible in one place.
    print('Saving demographic data')
    population_size = 23008366
    demographic_parameters = [age_p, gender_p, student_p, employment_p, job_p, family_size_dict, municipality_data,
                school_p, workplace_p, hospital_p, hospital_size_p, hospital_sizes, population_size]

    with open(save_path/Path('demographic_parameters.pkl'), 'wb') as f:
        pickle.dump(demographic_parameters, f)


def get_contact_p(save_path):
    """Search bounds of the contact block. Rewritten for Phase D -- the reasoning for every
    entry is in apply_phaseD_parameters.py, which applies the same values to an existing
    variable/ directory without regenerating the census data."""
    contact_parameters = {
        # [contact gate, contact-again-tomorrow, healthy daily p, symptomatic daily p,
        #  logistic steepness, symptom phase, width]

        # Household: the target is 1.86 close contacts per day (B25) against 2.8 other
        # members, which needs a daily probability near 0.66; the old ceiling was 0.5. The
        # symptomatic probability stays free because B19 exempts the household -- a sick
        # person really does stay home with the family.
        'household_lower_bound': [0.95, 0.4, 0.20, 0.01, 1, 0, 0],
        'household_upper_bound': [1, 1, 0.95, 0.90, 20, 10, 10],

        # School: symptomatic contact probability (index 3) is held at or below the healthy
        # one (index 2); B19 (a) also clamps it in the model itself.
        'school_lower_bound': [0.1, 0.5, 0.01, 0.001, 1, 0, 0],
        'school_upper_bound': [1, 1, 0.5, 0.01, 10, 10, 10],

        # Workplace: 1.49 contacts/day over the whole population is about 3/day for an
        # employed case, against a work group of about 10 people (B31).
        'workplace_lower_bound': [0.1, 0.5, 0.01, 0.001, 1, 0, 0],
        'workplace_upper_bound': [1, 1, 0.70, 0.05, 10, 10, 10],

        # Health care: floors lowered. The layer was generating 2.5 contacts per day for a
        # person who is not ill yet (finding E26); the clinic is meant to be visited after
        # symptoms start, which is what the symptomatic probability is for.
        'health_care_lower_bound': [0.0005, 0.5, 0.0002, 0.6, 1, 0, 0],
        'health_care_upper_bound': [0.008, 1, 0.006, 1, 10, 10, 10],

        # Municipality: index 0 is no longer a probability. The community layer draws
        # Poisson(nu * lambda) contacts (B27), so index 0 is lambda, the mean number of
        # distinct community contacts per case, and the city population no longer enters the
        # contact count (finding E4). The Taiwan tracing records give a median of 7.5 and a
        # mean of 87 friend/other contacts per index case; the heavy tail comes from nu.
        'municipality_lower_bound': [1.0, 0.5, 0.01, 0.001, 1, 0, 0],
        'municipality_upper_bound': [40.0, 1, 0.50, 0.50, 10, 10, 10],

        # Case-level infectiousness multiplier nu ~ Gamma(k, 1/k), capped (B17):
        # [k, cap]. k below 1 is where superspreading lives; Taiwan's own offspring
        # dispersion is k = 0.29.
        'overdispersion_lower_bound': [0.05, 3.0],
        'overdispersion_upper_bound': [2.0, 60.0]
    }
    print('Saving contact data')
    with open(save_path/Path('contact_parameters.pkl'), 'wb') as f:
        pickle.dump(contact_parameters, f)


def gamma_fit_bootstrap(days_data, CI=0.68, allow_negative=False):
    n_bootstrap = 1000
    bootstrap_alphas = np.zeros(n_bootstrap)
    bootstrap_locs = np.zeros(n_bootstrap)
    bootstrap_scales = np.zeros(n_bootstrap)
    for i in tqdm(range(n_bootstrap)):
        bootstrap_sample = np.random.choice(days_data, size=len(days_data), replace=True)
        alpha, loc, scale = stats.gamma.fit(bootstrap_sample)
        if loc < 0:
            if allow_negative:
                pass
            else:
                
                alpha, loc, scale = stats.gamma.fit(bootstrap_sample, floc=0)
        bootstrap_alphas[i] = alpha
        bootstrap_locs[i] = loc
        bootstrap_scales[i] = scale

    alpha = np.median(bootstrap_alphas)
    alpha_lb = np.percentile(bootstrap_alphas, (1 - CI) / 2 * 100)
    alpha_ub = np.percentile(bootstrap_alphas, (1 + CI) / 2 * 100)
    loc = np.median(bootstrap_locs)
    loc_lb = np.percentile(bootstrap_locs, (1 - CI) / 2 * 100)
    loc_ub = np.percentile(bootstrap_locs, (1 + CI) / 2 * 100)
    scale = np.median(bootstrap_scales)
    scale_lb = np.percentile(bootstrap_scales, (1 - CI) / 2 * 100)
    scale_ub = np.percentile(bootstrap_scales, (1 + CI) / 2 * 100)

    return alpha, alpha_lb, alpha_ub, loc, loc_lb, loc_ub, scale, scale_lb, scale_ub

def get_state_transition_gamma_p(course_of_disease_data_path, course_parameters, course_parameters_lb, course_parameters_ub):
    Taiwan_data_sheet = pd.read_excel(course_of_disease_data_path/'figshare_taiwan_covid.xlsx', 
                                  sheet_name=0)
    Taiwan_data_sheet.columns = Taiwan_data_sheet.columns.str.strip().str.lower().str.\
        replace(' ', '_').str.replace('(', '').str.replace(')', '')

    # Clean data
    data_1_to_579 = clean_taiwan_data(Taiwan_data_sheet, 1, 579)

    # Extract course of disease data
    asymptomatic_to_symptom_days = extract_state_data(
        data_1_to_579, 'earliest_infection_date', 'onset_of_symptom')
    asymptomatic_to_recover_days = extract_state_data(data_1_to_579, 'earliest_infection_date', 'recovery',
                                                    'onset_of_symptom')

    symptom_to_icu_days = extract_state_data(
        data_1_to_579, 'onset_of_symptom', 'icu')
    symptom_to_recover_days = extract_state_data(
        data_1_to_579, 'onset_of_symptom', 'recovery', 'icu')

    icu_to_recover_days = extract_state_data(
        data_1_to_579, 'icu', 'recovery')
    icu_to_dead_days = extract_state_data(
        data_1_to_579, 'icu', 'death_date')                                                 

    asymptomatic_to_confirmed_days = extract_state_data(
        data_1_to_579, 'earliest_infection_date', 'confirmed_date', 'onset_of_symptom')
    asymptomatic_to_confirmed_days = asymptomatic_to_confirmed_days.drop(asymptomatic_to_recover_days.index)
    symptomatic_to_confirmed_days = extract_state_data(
        data_1_to_579, 'onset_of_symptom', 'confirmed_date')
    symptomatic_to_confirmed_days = symptomatic_to_confirmed_days.drop(symptom_to_icu_days.index)
    symptomatic_to_confirmed_days = symptomatic_to_confirmed_days.drop(symptom_to_recover_days.index)

    # Symptom onset to confirmed
    if type(symptomatic_to_confirmed_days):
        symptomatic_to_confirmed_days = symptomatic_to_confirmed_days.apply(lambda x: x.days).to_numpy()

    CI = 0.68
    print(f'Fitting gamma distribution for symptomatic to confirmed days (data points: {len(symptomatic_to_confirmed_days)})')
    alpha, alpha_lb, alpha_ub, loc, loc_lb, loc_ub, scale, scale_lb, scale_ub = gamma_fit_bootstrap(symptomatic_to_confirmed_days, CI)
    course_parameters = np.append(course_parameters, (alpha, scale, loc))
    course_parameters_lb = np.append(course_parameters_lb, (alpha_lb, scale_lb, loc_lb))
    course_parameters_ub = np.append(course_parameters_ub, (alpha_ub, scale_ub, loc_ub))

    # Asymptomatic to recovered
    # NOTE: Since we only have 2 data points, there is no point to do bootstraping. We just use the Gamma fit

    if type(asymptomatic_to_recover_days) != np.ndarray:
        asymptomatic_to_recover_days = asymptomatic_to_recover_days.apply(lambda x: x.days).to_numpy()
    
    print('Fitting gamma distribution for asymptomatic to recover days (data points: {})'.format(len(asymptomatic_to_recover_days)))
    alpha, loc, scale = stats.gamma.fit(asymptomatic_to_recover_days, floc=0)
    alpha_lb = alpha*0.8
    alpha_ub = alpha*1.2
    loc_lb = loc
    loc_ub = loc
    scale_lb = scale*0.8
    scale_ub = scale*1.2
    course_parameters = np.append(course_parameters, (alpha, scale, loc))
    course_parameters_lb = np.append(course_parameters_lb, (alpha_lb, scale_lb, loc_lb))
    course_parameters_ub = np.append(course_parameters_ub, (alpha_ub, scale_ub, loc_ub))

    # Symptomatic to critically ill
    if type(symptom_to_icu_days) != np.ndarray:
        symptom_to_icu_days = symptom_to_icu_days.apply(lambda x: x.days).to_numpy()

    print('Fitting gamma distribution for symptomatic to critically ill days (data points: {})'.format(len(symptom_to_icu_days)))
    alpha, alpha_lb, alpha_ub, loc, loc_lb, loc_ub, scale, scale_lb, scale_ub = gamma_fit_bootstrap(symptom_to_icu_days, CI, allow_negative=True)
    course_parameters = np.append(course_parameters, (alpha, scale, loc))
    course_parameters_lb = np.append(course_parameters_lb, (alpha_lb, scale_lb, loc_lb))
    course_parameters_ub = np.append(course_parameters_ub, (alpha_ub, scale_ub, loc_ub))

    # Symptomatic to recovered
    if type(symptom_to_recover_days) != np.ndarray:
        symptom_to_recover_days = symptom_to_recover_days.apply(lambda x: x.days).to_numpy()
    print('Fitting gamma distribution for symptomatic to recovered days (data points: {})'.format(len(symptom_to_recover_days)))
    alpha, loc, scale = stats.gamma.fit(symptom_to_recover_days)
    alpha_lb = alpha*0.8
    alpha_ub = alpha*1.2
    loc_lb = loc*0.8
    loc_ub = loc*1.1
    scale_lb = scale*0.8
    scale_ub = scale*1.2
    course_parameters = np.append(course_parameters, (alpha, scale, loc))
    course_parameters_lb = np.append(course_parameters_lb, (alpha_lb, scale_lb, loc_lb))
    course_parameters_ub = np.append(course_parameters_ub, (alpha_ub, scale_ub, loc_ub))

    # Critically ill to recovered
    if type(icu_to_recover_days) != np.ndarray:
        icu_to_recover_days = icu_to_recover_days.apply(lambda x: x.days).to_numpy()
    print('Fitting gamma distribution for critically ill to recovered days (data points: {})'.format(len(icu_to_recover_days)))
    alpha, loc, scale = stats.gamma.fit(icu_to_recover_days)
    # alpha, alpha_lb, alpha_ub, loc, loc_lb, loc_ub, scale, scale_lb, scale_ub = gamma_fit_bootstrap(icu_to_recover_days, CI)
    alpha_lb = alpha*0.8
    alpha_ub = alpha*1.2
    loc_lb = loc*0.8
    loc_ub = loc*1.2
    scale_lb = scale*0.8
    scale_ub = scale*1.2
    course_parameters = np.append(course_parameters, (alpha, scale, loc))
    course_parameters_lb = np.append(course_parameters_lb, (alpha_lb, scale_lb, loc_lb))
    course_parameters_ub = np.append(course_parameters_ub, (alpha_ub, scale_ub, loc_ub))

    # Asymptomatic to death
    # Parameters from "Ward, T., & Johnsen, A. (2021). Understanding an evolving pandemic: An analysis of the clinical time delay distributions of COVID-19 in the United Kingdom. Plos one, 16(10), e0257978."
    infection_to_death_shape = 3.07
    infection_to_death_shape_lb = 2.96
    infection_to_death_shape_ub = 3.17
    infection_to_death_scale = 1/0.15
    infection_to_death_scale_lb = 6.25
    infection_to_death_scale_ub = 6.67
    course_parameters = np.append(course_parameters, (infection_to_death_shape, infection_to_death_scale))
    course_parameters_lb = np.append(course_parameters_lb, (infection_to_death_shape_lb, infection_to_death_scale_lb))
    course_parameters_ub = np.append(course_parameters_ub, (infection_to_death_shape_ub, infection_to_death_scale_ub))

    # Negative test to positive test
    negative_test_date_1 = pd.to_timedelta(extract_state_data(
    data_1_to_579, 'confirmed_date', 'negative_test_date_1'))
    negative_test_date_2 = pd.to_timedelta(extract_state_data(
        data_1_to_579, 'confirmed_date', 'negative_test_date_2'))
    negative_test_date_3 = pd.to_timedelta(extract_state_data(
        data_1_to_579, 'confirmed_date', 'negative_test_date_3'))
    negative_test_date_4 = pd.to_timedelta(extract_state_data(
        data_1_to_579, 'confirmed_date', 'negative_test_date_4'))

    negative_test_date_array = np.hstack([negative_test_date_1.dt.days.to_numpy(), negative_test_date_2.dt.days.to_numpy(
    ), negative_test_date_3.dt.days.to_numpy(), negative_test_date_4.dt.days.to_numpy()])
    print('Fitting gamma distribution for negative test to positive test days (data points: {})'.format(len(negative_test_date_array)))
    alpha, alpha_lb, alpha_ub, loc, loc_lb, loc_ub, scale, scale_lb, scale_ub = gamma_fit_bootstrap(-negative_test_date_array, CI)
    course_parameters = np.append(course_parameters, (alpha, scale, loc))
    course_parameters_lb = np.append(course_parameters_lb, (alpha_lb, scale_lb, loc_lb))
    course_parameters_ub = np.append(course_parameters_ub, (alpha_ub, scale_ub, loc_ub))


    return(course_parameters, course_parameters_lb, course_parameters_ub, icu_to_recover_days, icu_to_dead_days)

def get_daily_secondary_attack_rate(course_parameters, course_parameters_lb, course_parameters_ub):
    """Append the five layers' 25 daily attack rates and their bounds.

    The anchors, the Ge 2021 profile and the cumulative-to-daily conversion all live in
    sar_anchors.py so that this file and apply_phaseD_parameters.py cannot disagree; they
    used to carry two different sets of contact-day constants for the same quantity, and
    only the second of them ever reached variable/course_parameters_*.npy.
    """
    from sar_anchors import LAYERS, attack_rate_block
    for layer in LAYERS:
        rate, rate_lb, rate_ub = attack_rate_block(layer)
        course_parameters = np.append(course_parameters, rate)
        course_parameters_lb = np.append(course_parameters_lb, rate_lb)
        course_parameters_ub = np.append(course_parameters_ub, rate_ub)
    return (course_parameters, course_parameters_lb, course_parameters_ub)


def _get_daily_secondary_attack_rate_published(course_parameters, course_parameters_lb, course_parameters_ub):
    """The pre-Phase-D version, kept for reference only. See finding E52 for what is wrong
    with it: the layer rates below are CUMULATIVE per contact but were applied once per
    contact DAY, the household bounds were never rescaled with the centre, and three of the
    five central values appear in neither of the two cited papers."""
    # Cheng, Hao-Yuan, et al. "Contact tracing assessment of COVID-19 transmission dynamics in Taiwan and risk at different exposure periods before and after symptom onset." JAMA internal medicine 180.9 (2020): 1156-1163.
    # Ge, Yang, et al. "COVID-19 transmission dynamics among close contacts of index patients with COVID-19: a population-based cohort study in Zhejiang Province, China." JAMA Internal Medicine 181.10 (2021): 1343-1350.
    attack_rate = np.array([0.86, 0.98, 1.09, 1.16, 1.16, 1.07, 0.95, 0.84, 0.78, 0.78, 0.86,
                       1.02, 1.18, 1.3, 1.34, 1.33, 1.27, 1.19, 1.11, 1.05, 1.01, 0.99, 0.98, 0.99, 0.99]) # Relative risk
    shift = np.mean(attack_rate) -1 # Mean of the relative risk should be 1
    attack_rate = attack_rate - shift
    attack_rate_lb = np.array([0.39, 0.63, 0.9, 0.91, 0.81, 0.75, 0.72, 0.7, 0.65, 0.63, 0.72, 
                            0.9, 1.05, 1.13, 1.18, 1.19, 1.15, 1.07, 0.98, 0.92, 0.89, 0.88, 0.84, 0.76, 0.69]) - shift
    attack_rate_ub = np.array([1.91, 1.52, 1.31, 1.48, 1.66, 1.54, 1.26, 1.02, 0.94, 0.96, 1.04,
                            1.15, 1.34, 1.49, 1.54, 1.49, 1.4, 1.32, 1.26, 1.2, 1.14, 1.11, 1.15, 1.27, 1.43]) - shift


    # The literature secondary attack rates below are CUMULATIVE per close contact over the
    # whole exposure window, but this vector is applied once per contact DAY inside
    # Draw_contact_data. Using a cumulative SAR as a daily probability overestimates
    # transmission by roughly the number of contact days; measured on the previous run
    # (cheng2020 mode, 10,000 index cases) the mean number of contact days per contact is
    # household 2.00, school 2.95, workplace 4.02, health care 1.99, municipality 3.91.
    # Convert with p_daily = 1 - (1 - SAR)**(1/n_days).
    contact_days = {'household': 2.00, 'school': 2.95, 'workplace': 4.02,
                    'health_care': 1.99, 'municipality': 3.91}

    def to_daily(cumulative_sar, layer):
        return 1 - (1 - cumulative_sar) ** (1 / contact_days[layer])

    # (lower bound, central estimate, upper bound) CUMULATIVE secondary attack rate.
    #   household   : Cheng 2020 all-infection 10/151 = 6.6%; lb = Cheng clinical
    #                 (10-3)/151 = 4.6%, ub = Huang 2021 pooled Taiwan 10.1%. The
    #                 all-infection figure is used because CovSyn counts every infection.
    #   health care : Cheng 2020 all-infection 6/698 = 0.86%.
    #   school / workplace / municipality: layer-specific literature, because Cheng's
    #                 "Others" category (1836 contacts, 1 case) pools casual community
    #                 contacts and is not comparable to repeated school/workplace contact.
    # workplace: the upper bound used to be 15%, more than four times the literature
    # estimate of 3.4%, and the optimizer duly pushed the workplace attack rate to ~7.6%
    # cumulative. Because workplace contacts are concentrated in ages 20-59 (44% / 49%),
    # that inflated the 20-39 reference group and dragged the POOLED 60+ age risk ratio
    # down to 0.81 against an input of 1.32 (a Simpson effect: the within-layer ratios were
    # reproduced correctly). The bound is tightened to 5%, ~1.5x the literature estimate,
    # matching the relative width used for the household layer.
    layer_sar = {'household':    (0.046, 0.0662, 0.101),
                 'school':       (0.010, 0.023,  0.040),
                 'workplace':    (0.015, 0.034,  0.050),
                 'health_care':  (0.001, 0.0086, 0.016),
                 'municipality': (0.001, 0.002,  0.010)}

    for layer in ('household', 'school', 'workplace', 'health_care', 'municipality'):
        sar_lb, sar, sar_ub = layer_sar[layer]
        rate    = attack_rate    * (to_daily(sar,    layer) / np.mean(attack_rate))
        rate_lb = attack_rate_lb * (to_daily(sar_lb, layer) / np.mean(attack_rate_lb))
        rate_ub = attack_rate_ub * (to_daily(sar_ub, layer) / np.mean(attack_rate_ub))
        # The Ge et al. lower/upper profiles have a different shape from the central one,
        # so clip the centre into the box to keep the initial guess feasible.
        rate = np.clip(rate, rate_lb, rate_ub)
        course_parameters = np.append(course_parameters, rate)
        course_parameters_lb = np.append(course_parameters_lb, rate_lb)
        course_parameters_ub = np.append(course_parameters_ub, rate_ub)

    return(course_parameters, course_parameters_lb, course_parameters_ub)

def check_course_of_disease_boundary(course_parameters, course_parameters_lb, course_parameters_ub):
    # Check if course_parameters stay within course_parameters_lb and course_parameters_ub
    within_bounds = np.logical_and(course_parameters >= course_parameters_lb, 
                                course_parameters <= course_parameters_ub)

    if np.all(within_bounds):
        print("All course_parameters are within their respective bounds.")
    else:
        out_of_bounds = np.where(~within_bounds)[0]
        print(f"Warning: {len(out_of_bounds)} parameter(s) are out of bounds:")
        for idx in out_of_bounds:
            print(f"Parameter {idx}: {course_parameters[idx]:.4f} "
                f"(bounds: {course_parameters_lb[idx]:.4f}, {course_parameters_ub[idx]:.4f})")



def get_epidemiolocial_parameters(course_of_disease_data_path, save_path):
    course_parameters = np.array([])
    course_parameters_lb = np.array([])
    course_parameters_ub = np.array([])

    # Latent period
    # Xin, Hualei, et al. "Estimating the latent period of coronavirus disease 2019 (COVID-19)." Clinical Infectious Diseases 74.9 (2022): 1678-1681.
    # Latent period targeted at the LOWER edge of the literature reported-mean range
    # (4.1-5.5 d, Xin et al. 2022) -- decision D1 / option A: latent is brought inside the
    # reported range and the generation time is allowed to sit above its reported range,
    # because the model enforces infection >= latent so E[generation] >= E[latent] and the
    # two literature ranges (latent >= 4.1, generation <= 5.2) cannot both be satisfied.
    # Shape is fixed so realised mean = shape*scale stays within bounds, and latent
    # mean_ub (4.5) < incubation mean_lb (5.2).
    latent_period_shape = 4
    latent_period_shape_lb = latent_period_shape
    latent_period_shape_ub = latent_period_shape
    latent_period_mean = 4.3
    latent_period_mean_lb = 4.1
    latent_period_mean_ub = 4.5
    latent_period_scale = latent_period_mean/latent_period_shape
    latent_period_scale_lb = latent_period_mean_lb/latent_period_shape
    latent_period_scale_ub = latent_period_mean_ub/latent_period_shape
    course_parameters = np.append(course_parameters, (latent_period_shape, latent_period_scale))
    course_parameters_lb = np.append(course_parameters_lb, (latent_period_shape_lb, latent_period_scale_lb))
    course_parameters_ub = np.append(course_parameters_ub, (latent_period_shape_ub, latent_period_scale_ub))

    # Infectious period
    # Sanche, Steven, et al. "High contagiousness and rapid spread of severe acute respiratory syndrome coronavirus 2." Emerging infectious diseases 26.7 (2020): 1470.
    infectious_period_shape = 4
    # Fix the shape so that the scale bounds (which are derived from a fixed shape)
    # are consistent with the shape actually used during optimization. Otherwise the
    # optimizer can pick shape_ub * scale_ub = 6 * 3.5 = 21 days, far above the
    # intended mean upper bound of 14 days. With the shape fixed, the realised mean
    # equals shape * scale and stays within [mean_lb, mean_ub] = [4, 14].
    infectious_period_shape_lb = infectious_period_shape
    infectious_period_shape_ub = infectious_period_shape
    infectious_period_mean = 8
    infectious_period_mean_lb = 4
    infectious_period_mean_ub = 14
    infectious_period_scale = infectious_period_mean/infectious_period_shape
    infectious_period_scale_lb = infectious_period_mean_lb/infectious_period_shape
    infectious_period_scale_ub = infectious_period_mean_ub/infectious_period_shape
    course_parameters = np.append(course_parameters, (infectious_period_shape, infectious_period_scale))
    course_parameters_lb = np.append(course_parameters_lb, (infectious_period_shape_lb, infectious_period_scale_lb))
    course_parameters_ub = np.append(course_parameters_ub, (infectious_period_shape_ub, infectious_period_scale_ub))

    # Incubation period
    # Cheng, Hao-Yuan, et al. "Contact tracing assessment of COVID-19 transmission dynamics in Taiwan and risk at different exposure periods before and after symptom onset." JAMA internal medicine 180.9 (2020): 1156-1163.
    # Incubation held inside the literature reported-mean range (3.9-8.0 d over 17 studies)
    # with the lower bound (5.2) above the latent mean upper bound (4.5), so incubation >
    # latent and a positive presymptomatic window survives the incubation>=latent
    # truncation in draw_incubation_period().
    # Shape is fixed so realised mean = shape * scale stays within [mean_lb, mean_ub].
    incubation_period_shape = 2
    incubation_period_shape_lb = incubation_period_shape
    incubation_period_shape_ub = incubation_period_shape
    incubation_period_mean = 5.6    # > latent mean_ub (4.5)
    incubation_period_mean_lb = 5.2
    incubation_period_mean_ub = 8.0
    incubation_period_scale = incubation_period_mean/incubation_period_shape
    incubation_period_scale_lb = incubation_period_mean_lb/incubation_period_shape
    incubation_period_scale_ub = incubation_period_mean_ub/incubation_period_shape
    course_parameters = np.append(course_parameters, (incubation_period_shape, incubation_period_scale))
    course_parameters_lb = np.append(course_parameters_lb, (incubation_period_shape_lb, incubation_period_scale_lb))
    course_parameters_ub = np.append(course_parameters_ub, (incubation_period_shape_ub, incubation_period_scale_ub))

    course_parameters, course_parameters_lb, course_parameters_ub, icu_to_recover_days, icu_to_dead_days = \
        get_state_transition_gamma_p(course_of_disease_data_path, course_parameters, course_parameters_lb, course_parameters_ub)

    # Age dependent risk ratio
    # Cheng, Hao-Yuan, et al. "Contact tracing assessment of COVID-19 transmission dynamics in Taiwan and risk at different exposure periods before and after symptom onset." JAMA internal medicine 180.9 (2020): 1156-1163.
    # Age-related risk ratio of the secondary attack rate, for age 0-19, 20-39, 40-59, 60+.
    # LOCKED (lb == ub): set by hand and swept by trial and error, not optimized.
    # Cheng et al. report a CLINICAL attack-rate ratio (asymptomatic secondary cases are
    # excluded from their numerator), but CovSyn counts every infection and its
    # asymptomatic fraction is age independent, so the comparable target is Cheng's
    # ALL-INFECTION ratio recomputed from his own counts:
    #     0-19   1/281  = 0.356%  -> 0.52
    #     20-39  8/1161 = 0.689%  -> 1     (reference)
    #     40-59 10/794  = 1.259%  -> 1.83
    #     60+    3/331  = 0.906%  -> 1.32
    # The 0-19 value 0.5 is independently corroborated by the susceptibility literature
    # (Zhang 0.34, Davies 0.40, Viner 0.56, Uthman wild-type 0.58, Madewell 0.59);
    # see covsyn_age_weight_litreview.md. 20-39 stays at 1 as the readability anchor --
    # after the per-layer /Z normalisation in Data_synthesize.py the vector only fixes the
    # relative shape, so one group must be pinned to remove the scale degeneracy.
    # Phase D uses the B14 trial value: the input vector is not the target vector, because
    # the per-layer normalisation and the different age composition of each layer shift the
    # measured ratios. [0.39, 1, 1.90, 1.44] measured [0.557, 1, 1.817, 1.313] against the
    # target [0.52, 1, 1.83, 1.32]. The layers changed again in Phase D, so this has to be
    # re-measured with rr_exact.py after the rerun and re-tuned if it has drifted.
    age_risk_ratios = np.array([0.39, 1, 1.90, 1.44])
    age_risk_ratios_lb = age_risk_ratios.copy()
    age_risk_ratios_ub = age_risk_ratios.copy()
    course_parameters = np.append(course_parameters, age_risk_ratios)
    course_parameters_lb = np.append(course_parameters_lb, age_risk_ratios_lb)
    course_parameters_ub = np.append(course_parameters_ub, age_risk_ratios_ub)

    # Natural immunity
    # Pilz, S., Chakeri, A., Ioannidis, J. P., Richter, L., Theiler‐Schwetz, V., Trummer, C., ... & Allerberger, F. (2021). SARS‐CoV‐2 re‐infection risk in Austria. European journal of clinical investigation, 51(4), e13520.
    natural_immunity_rate = 0.91  
    natural_immunity_rate_lb = 0.87
    natural_immunity_rate_ub = 0.93
    course_parameters = np.append(course_parameters, natural_immunity_rate)
    course_parameters_lb = np.append(course_parameters_lb, natural_immunity_rate_lb)
    course_parameters_ub = np.append(course_parameters_ub, natural_immunity_rate_ub)

    # Vaccine efficacy
    # Article_Dhayfule2021_Vaccine_review_210514
    vaccine_rate = 0 # randomly set
    vaccine_rate_lb = 0
    vaccine_rate_ub = 0
    vaccine_efficacy = 0.905
    vaccine_efficacy_lb = 0.881
    vaccine_efficacy_ub = 0.924
    course_parameters = np.append(course_parameters, (vaccine_rate, vaccine_efficacy))
    course_parameters_lb = np.append(course_parameters_lb, (vaccine_rate_lb, vaccine_efficacy_lb))
    course_parameters_ub = np.append(course_parameters_ub, (vaccine_rate_ub, vaccine_efficacy_ub))

    course_parameters, course_parameters_lb, course_parameters_ub = get_daily_secondary_attack_rate(course_parameters, course_parameters_lb, course_parameters_ub)

    # Transition probability
    shift_percentage = 0.3
    # B33: the whole severity cascade now comes from the Taiwan contact-tracing file itself
    # (579 cases), so the three transition probabilities are consistent with each other and
    # their product reproduces the observed case fatality of 1.2%:
    #   asymptomatic 23.7% (137/579 with no onset date)   -> infection_to_recovered
    #   symptomatic -> ICU 12.7% (56/442)                 -> 1 - symptom_to_recovered
    #   ICU -> death 12.5% (7/56)                         -> 1 - critically_ill_to_recovered
    # The previous asymptomatic share (9%, Cheng) and ICU share (18%, a news report with no
    # traceable denominator, finding E28) are replaced.
    infection_to_recovered_transition_p = 0.237
    symptom_to_recovered_transition_p = 1-0.127
    print('Number of critically_ill_to_recovered_transition_p data points: {}'.format(len(icu_to_recover_days)+len(icu_to_dead_days)))
    critically_ill_to_recovered_transition_p = len(icu_to_recover_days)/(
        len(icu_to_recover_days)+len(icu_to_dead_days))

    transition_p = np.array([infection_to_recovered_transition_p, symptom_to_recovered_transition_p,
                            critically_ill_to_recovered_transition_p])
    transition_p_lb = transition_p-transition_p*shift_percentage
    transition_p_lb[transition_p_lb<0] = 0
    transition_p_ub = transition_p+transition_p*shift_percentage
    transition_p_ub[transition_p_ub>1] = 1

    course_parameters = np.hstack([course_parameters, transition_p])
    course_parameters_lb = np.hstack([course_parameters_lb, transition_p_lb])
    course_parameters_ub = np.hstack([course_parameters_ub, transition_p_ub])

    check_course_of_disease_boundary(course_parameters, course_parameters_lb, course_parameters_ub)
    print('Saving course of disease parameters')
    with open('./variable/course_parameters.npy', 'wb') as f:
        np.save(f, course_parameters)
    with open('./variable/course_parameters_lb.npy', 'wb') as f:
        np.save(f, course_parameters_lb)
    with open('./variable/course_parameters_ub.npy', 'wb') as f:
        np.save(f, course_parameters_ub)


if __name__ == "__main__":
    start_time = time.time()
    # Path
    demographic_data_path = Path('./data/demographic_data')
    course_of_disease_data_path = Path('./data/structured_course_of_disease_data')
    save_path = Path('./variable')

    get_demographic_data(demographic_data_path, save_path)
    get_contact_p(save_path)
    get_epidemiolocial_parameters(course_of_disease_data_path, save_path)


    print(f'--- Done {(time.time() - start_time):.2f} seconds ---')