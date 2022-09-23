import bs4
import requests
import pandas as pd

res = requests.get('https://liquipedia.net/valorant/VCT/2022/Champions/Statistics')
soup = bs4.BeautifulSoup(res.text,'lxml')
soup.select('table')

val_table = soup.find('table',attrs ={'class':'wikitable wikitable-striped sortable'})
val_table_data = val_table.tbody.find_all('tr')

headings=[]
for td in val_table_data[0].find_all('th'):
    headings.append(td.text)

map_table = soup.find("table",attrs={"class":"table table-bordered"})
map_table_values = map_table.tbody.find_all("tr")

final_data = []
for i in range(len(map_table_values)):
    temp=[]
    for td in map_table_values[i].find_all(['td','th']):
        temp.append(td.text)
    final_data.append(temp)
map_pick = pd.DataFrame(final_data)

map_pick.columns=map_pick.iloc[0]
map_pick.drop(0,inplace=True)
map_pick.set_index(keys='Map',inplace=True)

for i in map_pick.columns:
    map_pick[i]=map_pick[i].apply(lambda x:int(x[:-1]))

map_pick.to_csv('Data/map_pick_stats.csv')

side_table = soup.find("table",attrs={"class":"wikitable wikitable-bordered wikitable-striped"})
side_table_values = side_table.tbody.find_all("tr")

final_data = []
for i in range(len(side_table_values)):
    temp=[]
    for td in side_table_values[i].find_all(['td','th']):
        temp.append(td.text)
    final_data.append(temp)
side_pick = pd.DataFrame(final_data)
side_pick.columns=side_pick.iloc[0]
side_pick.drop(0,inplace=True)
side_pick.set_index(keys='Map',inplace=True)
side_pick['Atk Wins']=side_pick['Atk Wins'].apply(lambda x:int(x.split()[0]))
side_pick['Def Wins']=side_pick['Def Wins'].apply(lambda x:int(x.split()[0]))

side_pick.to_csv('Data/side_pick_stats.csv')


banned_table = soup.find("table",attrs={"class":"table table-bordered"})
banned_table = banned_table.find_next("table").find_next("table")
banned_table_values = banned_table.tbody.find_all("tr")


final_table=[]
for i in range(len(banned_table_values)):
    temp=[]
    for td in banned_table_values[i].find_all(['td','th']):
        temp.append(td.text)
    final_table.append(temp)
banned_final = pd.DataFrame(final_table)

banned_final.columns = banned_final.iloc[0]
banned_final.drop(0,inplace=True)
banned_final.set_index('Map',inplace=True)

for i in map_pick.columns:
    banned_final[i]=banned_final[i].apply(lambda x:int(x[:-1]))


banned_final.to_csv('Data/banned_maps_stats.csv')

data={'Player':[],'Country':[],'Team':[],'Agents':[]}
final_stat=[]
for i in range(1,len(val_table_data)):
    
    temp=[]
    stat_temp=[]
    
    for td in val_table_data[i].find_all('td'):
        stat_temp.append(td.text)
        
        for t in td.find_all('a'):
            
            temp.append(t.get('title'))
            
    data['Country'].append(temp[0])
    data['Player'].append(temp[1])
    data['Team'].append(temp[2])
    data['Agents'].append(temp[3:])
    final_stat.append(stat_temp)
    
stat=pd.DataFrame(final_stat)
stat.drop([0,1,2],axis=1,inplace=True)
stat.columns=['Maps', 'K', 'D', 'A', 'KD', 'KDA', 'ACS/Map', 'K/Map', 'D/Map', 'A/Map']
info=pd.DataFrame(data)
final_player_stats = pd.concat([info,stat], axis = 1)
final_player_stats.to_csv('Data/player_stats.csv',index=False)