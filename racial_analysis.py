import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import numpy as np
import sklearn
import plotly.express as px
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import os 

#import plotly.express as px

covidDf = pd.read_csv('covid19.csv',parse_dates=['cdc_report_dt'])
poplulationDist = pd.read_csv('Census_Population.csv')
covidDf = covidDf.rename(columns={'cdc_report_dt':"report_date",'Race and ethnicity (combined)':"race"})
covidDf = covidDf.drop(['pos_spec_dt','onset_dt',"current_status","sex","age_group",'hosp_yn','icu_yn','death_yn','medcond_yn'], axis=1)

race_inc_df = pd.read_csv('IncomeByRace.csv')
race_inc_df.dropna(inplace = True)
def shahzad():
	raceList = ["White","Hispanic/Latino","Black","Asian","AIAN","NHPI"]
	racialDfDict = {}
	countByDayByRaceDf = pd.DataFrame()
	def raceLineGraphs(covidDf,raceName):
		raceDict = {k: v for k, v in covidDf.groupby('race')}
		race = pd.DataFrame.from_dict(raceDict[raceName])
		race_count = pd.DataFrame(race['report_date'].value_counts().rename_axis('Dates').to_frame(raceName+""))
		return race_count
	
	for race in raceList:
		countByDayByRaceDf = pd.concat([raceLineGraphs(covidDf,race),countByDayByRaceDf],axis=1)

	countByDayByRaceDf.reset_index(level=0,inplace=True)
	heapMap = sns.heatmap(countByDayByRaceDf.corr(),annot = True, vmin=-1, vmax=1, center= 0)
	plt.title("correlation on various races")
	plt.show()

	coeffDict = {}
	for race in raceList:
		countByDayByRaceDf['Date_Ordinal'] = pd.to_datetime(countByDayByRaceDf['Dates'])
		countByDayByRaceDf['Date_Ordinal'] =countByDayByRaceDf['Date_Ordinal'].map(dt.datetime.toordinal)

		linearAnalysisDf = pd.DataFrame({
			"Date_Ordinal":countByDayByRaceDf['Date_Ordinal'],
			race:countByDayByRaceDf[race]
			})
		linearAnalysisDf.dropna(inplace = True)
		y = np.asarray(linearAnalysisDf[race])
		X = linearAnalysisDf[['Date_Ordinal']]
		model = LinearRegression() #create linear regression object
		model.fit(X, y) #train model on train data
		Y_pred = model.predict(X)
		plt.scatter(X, y)
		plt.plot(X, Y_pred, color='red')
		plt.title("Cases and Regression analysis of " + str(race) + " individuals")
		plt.show()
		
		coeffDict[race] = model.score(X,y)

	########################
	keys = coeffDict.keys()
	vals = coeffDict.values()
	plt.bar(keys,vals)
	plt.title("Linear Regression Coefficeints")
	plt.show()
	########################

	def raceBarchart(covidDf,poplulationDist):
		racialDf= pd.DataFrame({
			'Cases':covidDf['race'].value_counts().rename_axis("Race")
		})
		racialDf= pd.merge(racialDf, poplulationDist, on="Race")
		racialDf = pd.merge(racialDf, race_inc_df, on="Race")
		Total_Cases = racialDf['Cases'].sum()
		racialDf['Case Percentage'] = racialDf['Cases']/Total_Cases
		racialDf['Percentage Of Race infected'] = (racialDf['Cases']/racialDf['population'])
		racialDf=racialDf.drop(['Population estimates, July 1, 2019'] , axis =1)
		#################################################
		plt.bar(racialDf['Race'],racialDf['Cases'])
		plt.xlabel("Race", fontsize=10)
		plt.xticks(rotation=45)
		plt.xticks(fontsize = 10)
		plt.ylabel("Cases ", labelpad=14)
		plt.title("Cases of covid19 per race", y=1.02);
		plt.show()
		#################################################
		plt.bar(racialDf['Race'],racialDf['Percentage Of Race infected'])
		plt.xlabel("Race", fontsize=10)
		plt.xticks(rotation=45)
		plt.xticks(fontsize = 10)
		plt.ylabel("Percent ", labelpad=14)
		plt.title("Percentage of Race infected by Race", y=1.02);
		plt.show()


		return racialDf
	#####################################
	return raceBarchart(covidDf,poplulationDist)

racialDf = shahzad()

def george(racialDf):

	race_inc_df = pd.read_csv('IncomeByRace.csv')
	race_inc_df.dropna(inplace = True)


	inc_race = race_inc_df
	plt.bar(inc_race['Race'],inc_race['Median Income'])
	plt.xlabel("Race", fontsize = 12)
	plt.ylabel('Avg Income ($)', fontsize = 12)
	plt.xticks(rotation=45)
	plt.title("Income by Race")
	plt.ylim(0,80000)
	plt.show()

	heatMapDf = racialDf.drop(['Percentage','population','Case Percentage',"Cases"],axis = 1)
	Heat_Map = sns.heatmap(heatMapDf.corr(), annot = True ,vmin=-1, vmax=1, center= 0)
	plt.show()

george(racialDf)

########################### Matt's Code ###############################

file_to_load = "mattcovid19.csv"

covid_data = pd.read_csv(file_to_load)


#Turn CSV to data frme#
df = covid_data

#Over View Race and Ethnicity and sex

covid_data = pd.read_csv(file_to_load)

covid_data_RE = covid_data.groupby(['Race and ethnicity (combined)', 'sex']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data

#Turn CSV to data frme#
long_df = covid_data_RE

fig = px.bar(long_df, x="Race and ethnicity (combined)", y="current_status_count", color="sex",)

fig.update_layout(
    title="Overview R&E Sex ",
    xaxis_title="R&E",
    yaxis_title="Number Of Cases",
    legend_title="Sex")

fig.show()
	 
#Overview Age Group

covid_data = pd.read_csv(file_to_load)


covid_data_age = covid_data.groupby(['Race and ethnicity (combined)', 'sex', 'age_group']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data_age

long_df = covid_data_age

fig = px.bar(long_df, x="age_group", y="current_status_count", color="sex",)

fig.update_layout(
    title="Overview R&E Sex",
    xaxis_title="Age Group",
    yaxis_title="Number Of Cases",
    legend_title="Sex")

fig.show()

#Overview Male and Female
covid_data = pd.read_csv(file_to_load)


covid_data_MF = covid_data.groupby(['Race and ethnicity (combined)', 'sex']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data_MF

long_df = covid_data_MF

fig = px.histogram(long_df, x="sex", y="current_status_count" , color="sex")

fig.update_layout(
    title="Overview Sex",
    xaxis_title="Sex",
    yaxis_title="Number Of Cases",
    legend_title="Sex")
        
fig.show()
	
#Filtered for Race AIAN

covid_data = pd.read_csv(file_to_load)


covid_data_AIAN = covid_data.loc[covid_data['Race and ethnicity (combined)'].isin(['AIAN'])]
covid_data_AIAN = covid_data_AIAN.groupby(['age_group', 'sex']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data_AIAN

long_df= covid_data_AIAN

fig = px.bar(long_df, x="sex", y="current_status_count",
             color='age_group', barmode='group',
             height=400)

fig.update_layout(
    title="Overview Sex and Race AIAN",
    xaxis_title="Sex",
    yaxis_title="Number Of Cases",
    legend_title="Sex")

fig.show()

#Age Group Race Asian
covid_data = pd.read_csv(file_to_load)

covid_data_ASIAN = covid_data.loc[covid_data['Race and ethnicity (combined)'].isin(['Asian'])]
covid_data_ASIAN = covid_data_ASIAN.groupby(['age_group', 'sex']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data_ASIAN

long_df= covid_data_ASIAN

fig = px.bar(long_df, x="sex", y="current_status_count",
             color='age_group', barmode='group',
             height=400)

fig.update_layout(
    title="Overview Sex and Race Asian",
    xaxis_title="Sex",
    yaxis_title="Number Of Cases",
    legend_title="Sex")


fig.show()

#Filtered For Race Black
covid_data = pd.read_csv(file_to_load)

covid_data_Black = covid_data.loc[covid_data['Race and ethnicity (combined)'].isin(['Black'])]
covid_data_Black = covid_data_Black.groupby(['age_group', 'sex']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data_Black

long_df= covid_data_Black

fig = px.bar(long_df, x="sex", y="current_status_count",
             color='age_group', barmode='group',
             height=400)

fig.update_layout(
    title="Overview Sex and Race Black",
    xaxis_title="Sex",
    yaxis_title="Number Of Cases",
    legend_title="Sex")
        
fig.show()

#Filtered for Hispanic/Latino
covid_data = pd.read_csv(file_to_load)

covid_data_Hispanic = covid_data.loc[covid_data['Race and ethnicity (combined)'].isin(['Hispanic/Latino'])]
covid_data_Hispanic = covid_data_Hispanic.groupby(['age_group', 'sex']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data_Hispanic

long_df= covid_data_Hispanic

fig = px.bar(long_df, x="sex", y="current_status_count",
             color='age_group', barmode='group',
             height=400)

fig.update_layout(
    title="Overview Sex and Race Hispanic/Latino",
    xaxis_title="Sex",
    yaxis_title="Number Of Cases",
    legend_title="Sex")
        
fig.show()

#Filtered for Multiple/Other
covid_data = pd.read_csv(file_to_load)

covid_data_Multiple = covid_data.loc[covid_data['Race and ethnicity (combined)'].isin(['Multiple/Other'])]
covid_data_Multiple = covid_data_Multiple.groupby(['age_group', 'sex']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data_Multiple

long_df= covid_data_Multiple

fig = px.bar(long_df, x="sex", y="current_status_count",
             color='age_group', barmode='group',
             height=400)

fig.update_layout(
    title="Overview Sex and Race Multiple/Other",
    xaxis_title="Sex",
    yaxis_title="Number Of Cases",
    legend_title="Sex")
        
fig.show()

#Filtered for NHPI
covid_data = pd.read_csv(file_to_load)

covid_data_NHPI = covid_data.loc[covid_data['Race and ethnicity (combined)'].isin(['NHPI'])]
covid_data_NHPI = covid_data_NHPI.groupby(['age_group', 'sex']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data_NHPI

long_df= covid_data_NHPI

fig = px.bar(long_df, x="sex", y="current_status_count",
             color='age_group', barmode='group',
             height=400)

fig.update_layout(
    title="Overview Sex and Race NHPI",
    xaxis_title="Sex",
    yaxis_title="Number Of Cases",
    legend_title="Sex")

fig.show()

#Filter for White
covid_data = pd.read_csv(file_to_load)

covid_data_White = covid_data.loc[covid_data['Race and ethnicity (combined)'].isin(['White'])]
covid_data_White = covid_data_White.groupby(['age_group', 'sex']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data_White

long_df= covid_data_White

fig = px.bar(long_df, x="sex", y="current_status_count",
             color='age_group', barmode='group',
             height=400)

fig.update_layout(
    title="Overview Sex and Race White",
    xaxis_title="Sex",
    yaxis_title="Number Of Cases",
    legend_title="Sex")

fig.show()

#Filter for unknown

covid_data = pd.read_csv(file_to_load)

covid_data_Unknown = covid_data.loc[covid_data['Race and ethnicity (combined)'].isin(['Unknown'])]
covid_data_Unknown = covid_data_Unknown.groupby(['age_group', 'sex']).agg(current_status_count=('current_status', 'count')).reset_index()
covid_data_Unknown

long_df= covid_data_Unknown

fig = px.bar(long_df, x="sex", y="current_status_count",
             color='age_group', barmode='group',
             height=400)

fig.update_layout(
    title="Overview Sex and Race Unknown",
    xaxis_title="Sex",
    yaxis_title="Number Of Cases",
    legend_title="Sex")

fig.show()

########################### Ken's Code ###############################
def Ken_Code():

	print('Ken Code Starts')
	# Create folder for .png files, if it does not exist
	Plot_Path = 'PNGFiles'
	if not os.path.exists(Plot_Path):
		os.makedirs(Plot_Path)

	SourceFilesFolder = 'SourceFiles'

	# Read Excel Files, skipthe first 3 rows for Contact, Exposure and Proximity files. 
	contact_file = pd.read_excel(SourceFilesFolder + '/Contact_With_Others.xls',skiprows = [0,1,2])
	exposure_file = pd.read_excel(SourceFilesFolder + '/Exposure_to_Disease_and_Infection.xls',skiprows = [0,1,2])
	p_proximity_file = pd.read_excel(SourceFilesFolder + '/Physical_Proximity.xls',skiprows = [0,1,2])
	occupation_file = pd.read_excel(SourceFilesFolder + '/Occupation Salary Population and  Type.xlsx')

	# Rename 'Occupation' field to 'OCCUPATION' for joining with Occupation file. 
	contact_file = contact_file.rename(columns={'Occupation': 'OCCUPATION'})

	print('Ken: Create OverallRiskScore_raw dataframe')
	# Join Contact, Exposure and Proximity data using Code column, 
	# let us call resulting dataset as 'OverallRiskScore'
	OverallRiskScore_raw = contact_file.merge(
		exposure_file[['E_D_I Context', 'Code']], on = 'Code').merge(
		p_proximity_file[['P_P Context', 'Code']], on = 'Code')

	# Add a new calculated column 'Overall Risk Score' with value = average of context field in all three files. 
	OverallRiskScore_raw['Overall Risk Score'] = round((
		OverallRiskScore_raw['C_W_ O Context'] + OverallRiskScore_raw['E_D_I Context'] + OverallRiskScore_raw['P_P Context'])/3, 2)

	# Re-order the column sequence for OverallRiskScore_raw dataset 
	OverallRiskScore_raw = OverallRiskScore_raw[['Code', 'OCCUPATION' , 'C_W_ O Context','E_D_I Context', 'P_P Context', 'Overall Risk Score' ]]

	# We need only two columns - OCCUPATION & Overall Risk Score, join this dataset with Occupation file. 
	OverallRiskScore_raw_reqCols = OverallRiskScore_raw[['OCCUPATION', 'Overall Risk Score']]

	print('Ken: Create OverallRiskScore dataframe')
	# Join OverallRiskScore_raw_reqCols dataset with Occupation File
	OverallRiskScore = OverallRiskScore_raw_reqCols.merge(occupation_file, on = 'OCCUPATION')

	# Add a new calculated field - PCT_TOT_EMP which is percentage of Total Emp for each with respect to total Employee count. 
	OverallRiskScore['PCT_TOT_EMP'] = round((OverallRiskScore['TOT_EMP']/OverallRiskScore['TOT_EMP'].sum())*100,2)

	# Some rows have * as mean salary, remove these records
	OverallRiskScore = OverallRiskScore[OverallRiskScore['MEAN_SALARY'] !='*']

	# Remove $ symbol from Mean Salary using Python Lambda function to consider mean salary as Integer value
	OverallRiskScore['MEAN_SALARY'] = OverallRiskScore['MEAN_SALARY'].apply(lambda x: str(x).replace('$', '')).astype('int64')

	print('Ken: Print Number of Rows for each dataset we have created')
	# Print Number of Rows for each dataset we have created
	print('******************************************************')
	print('Row Count of Contact File:- ' + str(len(contact_file)))
	print('Row Count of Exposure File:- ' + str(len(exposure_file)))
	print('Row Count of Physical Proximity File:- ' + str(len(p_proximity_file)))
	print('Row Count of Occupation File:- ' + str(len(occupation_file)))
	print('Row Count of OverallRiskScore_raw dataset:- ' + str(len(OverallRiskScore_raw)))
	print('Row Count of OverallRiskScore dataset:- ' + str(len(OverallRiskScore)))
	print('******************************************************')

	# Create folder for output files, if it does not exist
	Output_Path = 'OutputFiles'
	if not os.path.exists(Output_Path):
		os.makedirs(Output_Path)

	print('Ken: Save output files')
	# Set index = False to avoid exporting index column to csv file        
	OverallRiskScore_raw.to_csv(Output_Path + '/OverallRiskScore_raw.csv', index=False)
	OverallRiskScore.to_csv(Output_Path + '/OverallRiskScore.csv', index=False)

	print('Ken: Create Plot1')
	## Plot of Mean Salary to Overall Risk Score
	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	ax.scatter(OverallRiskScore['MEAN_SALARY'], OverallRiskScore['Overall Risk Score'],edgecolors='b')
	ax.set_xlabel('MEAN SALARY')
	ax.set_ylabel('Overall Risk Score')
	plt.grid(True)
	plt.title('MEAN SALARY vs. Overall Risk Score')
	# Save the plot as png file with all axis values and axis labels
	plt.savefig(Plot_Path + '/MeanSalary_OverallRiskScore.png', dpi=200, bbox_inches='tight')

	print('Ken: Create Plot2')
	## Plot of Mean Salary to Overall Risk Score
	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	OverallRiskScore_SalH = OverallRiskScore[OverallRiskScore['Health/Non Health']=='Health']
	OverallRiskScore_SalNH = OverallRiskScore[OverallRiskScore['Health/Non Health']=='Non Health']

	ax.scatter(OverallRiskScore_SalH['MEAN_SALARY'], OverallRiskScore_SalH['Overall Risk Score'],edgecolors='b')
	ax.scatter(OverallRiskScore_SalNH['MEAN_SALARY'], OverallRiskScore_SalNH['Overall Risk Score'],edgecolors='r')
	ax.set_xlabel('MEAN SALARY')
	ax.set_ylabel('Overall Risk Score')
	plt.legend(['Health', 'Non Health'])
	plt.grid(True)
	plt.title('MEAN SALARY vs. Overall Risk Score with Health and Non Health Occupation')
	# Save the plot as png file with all axis values and axis labels
	plt.savefig(Plot_Path + '/MeanSalary_OverallRiskScore_WithHealthVsNonHealth.png', dpi=200, bbox_inches='tight')
	
	print('Ken: Create Plot3')
	## Plot of Mean Salary to Overall Risk Score
	ORS_Cat = pd.read_csv('SourceFiles/OverallRiskScore_withCategory.csv')
	ORS_Cat.loc[(ORS_Cat['Category'].isnull()), 'Category'] = 'Others'
	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	ORS_Cat_Nurses = ORS_Cat[ORS_Cat['Category']=='Nurses']
	ORS_Cat_Doctors = ORS_Cat[ORS_Cat['Category']=='Doctors']
	ORS_Cat_Aides = ORS_Cat[ORS_Cat['Category']=='Aides']
	ORS_Cat_Others = ORS_Cat[ORS_Cat['Category']=='Others']

	ax.scatter(ORS_Cat_Nurses['MEAN_SALARY'], ORS_Cat_Nurses['Overall Risk Score'],edgecolors='b')
	ax.scatter(ORS_Cat_Doctors['MEAN_SALARY'], ORS_Cat_Doctors['Overall Risk Score'],edgecolors='r')
	ax.scatter(ORS_Cat_Aides['MEAN_SALARY'], ORS_Cat_Aides['Overall Risk Score'],edgecolors='g')
	ax.scatter(ORS_Cat_Others['MEAN_SALARY'], ORS_Cat_Others['Overall Risk Score'],edgecolors='y',marker='^')
	ax.set_xlabel('MEAN SALARY')
	ax.set_ylabel('Overall Risk Score')

	plt.grid(True)
	plt.title('MEAN SALARY vs. Top 20 Overall Risk Score with Category ')
	plt.legend(['Nurses', 'Doctors', 'Aides', 'Others'])
	# Save the plot as png file with all axis values and axis labels
	plt.savefig(Plot_Path + '/MeanSalary_Top20OverallRiskScore_WithCategory.png', dpi=200, bbox_inches='tight')

	print('Ken: Create Plot4')
	## Plot of Mean Salary to Overall Risk Score
	ORS_Cat = pd.read_csv('SourceFiles/OverallRiskScore_withCategory.csv')
	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	ORS_Health = ORS_Cat[ORS_Cat['Health/Non Health']=='Health']
	ORS_NonHealth = ORS_Cat[ORS_Cat['Health/Non Health']=='Non Health']

	ax.scatter(ORS_Health['MEAN_SALARY'], ORS_Health['Overall Risk Score'],edgecolors='r')
	ax.scatter(ORS_NonHealth['MEAN_SALARY'], ORS_NonHealth['Overall Risk Score'],edgecolors='y',marker='^')
	ax.set_xlabel('MEAN SALARY')
	ax.set_ylabel('Overall Risk Score')

	plt.grid(True)
	plt.title('MEAN SALARY vs. Overall Risk Score Health and Non Health Occupations')
	plt.legend(['Health', 'Non Health'])
	# Save the plot as png file with all axis values and axis labels
	plt.savefig(Plot_Path + '/MeanSalary_Health_NonHealth.png', dpi=200, bbox_inches='tight')

	print('Ken: Create Plot5')
	## Plot of Total Employees to Overall Risk Score
	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	ax.scatter(OverallRiskScore['TOT_EMP'], OverallRiskScore['Overall Risk Score'],edgecolors='b')
	ax.set_xlabel('TOT_EMP')

	# To stop matplotlib from showing exponential notation for Tot_Emp
	ax.ticklabel_format(useOffset=False, style='plain')
	ax.set_ylabel('Overall Risk Score')
	plt.grid(True)
	plt.title('TOT_EMP vs. Overall Risk Score')
	# Save the plot as png file with all axis values and axis labels
	plt.savefig(Plot_Path + '/TotEmp_OverallRiskScore.png', dpi=200, bbox_inches='tight')

	print('Ken: Create Plot6')
	## Plot of Total Employees to Overall Risk Score
	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	ax.scatter(OverallRiskScore['PCT_TOT_EMP'], OverallRiskScore['Overall Risk Score'],edgecolors='b')
	ax.set_xlabel('PCT_TOT_EMP')
	ax.set_ylabel('Overall Risk Score')
	plt.grid(True)
	plt.title('PCT_TOT_EMP vs. Overall Risk Score')
	# Save the plot as png file with all axis values and axis labels
	plt.savefig(Plot_Path + '/PctTotEmp_OverallRiskScore.png', dpi=200, bbox_inches='tight')

	print('Ken Code Ends')

Ken_Code()

#########################################################################