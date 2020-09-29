# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 08:53:57 2018
接matlab步骤sh_adt.m
读取.mat的SH与adt\sla的结果数据，然后进行数据分析
在SLA上绘制每对时刻的涡旋和SH位置

@author: zqr02
"""
#import netCDF4 as nc
import matplotlib.pyplot as plt
import netCDF4 as nc  
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import scipy.io
from matplotlib.patches import Ellipse, Circle
import math
from mpl_toolkits.basemap import Basemap
import skimage 
import os
"""
import urllib
import sys
from decimal import *
import numpy as np
import matplotlib 

import pylab
import pandas as pd
from matplotlib import animation
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
"""
argostarttime=datetime(1950,1,1)
eddystarttime=datetime(1900,1,1)
windstarttime=datetime(1900,1,1)
#-----------------function---------------------
def haversine(lon1, lat1, lon2, lat2): 
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """   
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])  
    #print 34
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2  
    c = 2 * math.atan(math.sqrt(a)/math.sqrt(1-a))   
    r = 6371 
    d=c * r
    #print type(d)
    return d
#-------------------code-----------------------------
    
argodata = scipy.io.loadmat('E:\\Python\\sh-adt-argo\\0.150.2\\SH_adtsla.mat')


#print (argodata['argotime'])
argolat=argodata['argolat']
argolon=argodata['argolon']
avisolat=argodata['avisolat']
avisolon=argodata['avisolon']
argotime=argodata['argotime']
avisotime=argodata['avisotime']
sla_d=argodata['diff_sla']
adt_d=argodata['diff_adt']
sh_d=argodata['diff_sh']
distance_all=[]
time_different_all=[]
sh_adt=[]
sh_sla=[]
windspeed=[]
dist_all_argo_aviso=[]
dist_all_argo_eddy=[]
aatime_different_all=[]
for i in range(0,len(argotime[0]),1):
    
    argotime_date= argostarttime+timedelta(argotime[0][i])
    #print (argotime_date.month)
    
    
    wind_file=nc.Dataset("F:\\data\\windspeed\\2014\\%s.nc"  %argotime_date.month)
    windtime=wind_file.variables['time'][:]
    windlon=wind_file.variables['longitude'][:]
    windlat=wind_file.variables['latitude'][:]
    windu=wind_file.variables['u10'][:]
    windv=wind_file.variables['v10'][:]

    for i_windtime in range(0,len(windtime)):
        windtime_data= windstarttime+timedelta(hours=float(windtime[i_windtime]))
        for i_windlon in range(len(windlon)):
            for i_windlat in range(len(windlat)):
                if abs((windtime_data-argotime_date).total_seconds() )< 10799 and abs(argolat[0][i]-windlat[i_windlat])<0.062 and abs(argolon[0][i]-windlon[i_windlon])<0.062:
                    wind_speed=math.sqrt(windv[i_windtime][i_windlat][i_windlon]*windv[i_windtime][i_windlat][i_windlon]+windu[i_windtime][i_windlat][i_windlon]*windu[i_windtime][i_windlat][i_windlon])
                    print (i,wind_speed)
                    windspeed.append(wind_speed)


for i in range(0,len(argotime[0]),2):
    print(i,argotime[0][i])
    lon_index=[]
    lat_index=[]
    slalon_index=[]
    slalat_index=[]
    hycomlat_plot=[]
    hycomlon_plot=[]
    slalat_plot=[]
    slalon_plot=[]
    argotime_date= argostarttime+timedelta(argotime[0][i])
    str_argotime=datetime.strftime(argotime_date,"%Y%m%d")
    print (str_argotime)
    eddydata=scipy.io.loadmat("F:\\data\\eddy\\eddies_aviso_2014_2014_select.mat") 
    eddytime=eddydata['time']
    eddylon=eddydata['lon']
    eddylat=eddydata['lat']
    eddyradiust=eddydata['Radius']
    eddyid=eddydata['ID']
    for i_eddy in range(len(eddytime)):
        #print (eddytime[i_eddy])
        eddytime_date= eddystarttime+timedelta(int(eddytime[i_eddy]))
        
        str_eddytime=datetime.strftime(eddytime_date,"%Y%m%d")
        if str_eddytime==str_argotime and eddyid[i_eddy]==977:
            i_sh=i//2
            print(1)
            distance_argo_eddy_a=haversine(argolon[0][i],argolat[0][i],eddylon[i_eddy], eddylat[i_eddy])#涡旋中心与argo距离
            distance_argo_eddy_b=haversine(argolon[0][i+1],argolat[0][i+1],eddylon[i_eddy], eddylat[i_eddy])
            dist_all_argo_eddy.append(distance_argo_eddy_a)
            dist_all_argo_eddy.append(distance_argo_eddy_b)
            distance=haversine(argolon[0][i],argolat[0][i],argolon[0][i+1],argolat[0][i+1])#argo两点间距离
            distance_all.append(distance)
            dist_argo_aviso_a=haversine(argolon[0][i],argolat[0][i],avisolon[0][i],avisolat[0][i])
            dist_argo_aviso_b=haversine(argolon[0][i+1],argolat[0][i+1],avisolon[0][i+1],avisolat[0][i+1])#每个argo和aviso之间距离
            dist_all_argo_aviso.append(dist_argo_aviso_a)
            dist_all_argo_aviso.append(dist_argo_aviso_b)
            time_diff=(argotime[0][i]-argotime[0][i+1])*24#argo两点间时间差距
            time_different_all.append(time_diff)
            aatimediff=abs(argotime[0][i]-avisotime[0][i])#Argo和AVISO时间只差
            aatime_different_all.append(aatimediff)
            sh_adt.append(100*(sh_d[0][i_sh]-adt_d[0][i_sh]))
            print(i_sh,sh_d[0][i_sh]-adt_d[0][i_sh],len(sh_adt))
            sh_sla.append(100*(sh_d[0][i_sh]-sla_d[0][i_sh]))
            radius=eddyradiust[i_eddy]/111000
            
            #--------------------------hycom的ssh底图设置-------------------------------
            minlat=int(min(argolat[0][i],argolat[0][i+1],eddylat[i_eddy]-radius))-0.2
            maxlat=math.ceil(max(argolat[0][i],argolat[0][i+1],eddylat[i_eddy]+radius))+0.2
            minlon=int(min(argolon[0][i],argolon[0][i+1],eddylon[i_eddy]-radius))-0.2
            maxlon=math.ceil(max(argolon[0][i],argolon[0][i+1],eddylon[i_eddy]+radius))+0.2
            '''hycom_data=nc.Dataset("E:\\data\\hycom\\%s\\%s00hycomdata.nc" %(str_argotime[0:4],str_argotime))
            ssh=hycom_data.variables['surf_el'][:]
            hycomlon=hycom_data.variables['lon'][:]
            hycomlat=hycom_data.variables['lat'][:]
            #print (minlat,maxlat,minlon,maxlon)
            for i_lon in range(len(hycomlon)):
                if minlon<hycomlon[i_lon]<maxlon:
                    lon_index.append(i_lon)
                    hycomlon_plot.append(hycomlon[i_lon])
            for i_lat in range(len(hycomlat)):
                if minlat<hycomlat[i_lat]<maxlat:
                    lat_index.append(i_lat)
                    hycomlat_plot.append(hycomlat[i_lat])
            x = hycomlon_plot
            y = hycomlat_plot
            lon_index_min=min(lon_index)
            lon_index_max=max(lon_index)+1
            lat_index_min=min(lat_index)
            lat_index_max=max(lat_index)+1
            ssh_matrix=np.mat(ssh)
            c= ssh_matrix[lat_index_min:lat_index_max:1,lon_index_min:lon_index_max:1]
            #print (ssh_matrix,lon_index_min,lon_index_max,1,lat_index_min,lat_index_max,1,c)
            # 生成网格数据'''
            #---------------------aviso的sla底图-----------------------------
            sla_data=nc.Dataset("F:\\data\\AVISO\\madt\\%s\\dt_global_allsat_phy_l4_%s_20170110.nc" %(str_argotime[0:4],str_argotime))
            sla=sla_data.variables['sla'][:]
            slalon=sla_data.variables['longitude'][:]
            slalat=sla_data.variables['latitude'][:]

            #print (minlat,maxlat,minlon,maxlon)
            for i_slalon in range(len(slalon)):
                if minlon<slalon[i_slalon]<maxlon:
                    slalon_index.append(i_slalon)
                    slalon_plot.append(slalon[i_slalon])
            for i_slalat in range(len(slalat)):
                if minlat<slalat[i_slalat]<maxlat:
                    slalat_index.append(i_slalat)
                    slalat_plot.append(slalat[i_slalat])
            x = slalon_plot
            y = slalat_plot
            slalon_index_min=min(slalon_index)
            slalon_index_max=max(slalon_index)+1
            slalat_index_min=min(slalat_index)
            slalat_index_max=max(slalat_index)+1
            sla_matrix=np.mat(sla)
            c= sla_matrix[slalat_index_min:slalat_index_max:1,slalon_index_min:slalon_index_max:1]
            #print (ssh_matrix,lon_index_min,lon_index_max,1,lat_index_min,lat_index_max,1,c)
            
            #--------------------------------------------------------------------
            X, Y = np.meshgrid(x, y)
            
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cir1 =Circle(xy = (eddylon[i_eddy], eddylat[i_eddy]), radius=radius, linewidth=5.0, alpha=0.1,color= 'green')#画涡旋  color='green'
            ax.add_patch(cir1)
            #plt.Circle(xy = (0.0, 0.0), radius=2, alpha=0.5)
            ax.plot(argolon[0][i],argolat[0][i],'b+')
            ax.plot(argolon[0][i+1],argolat[0][i+1],'b+')
            ax.plot(avisolon[0][i],avisolat[0][i],'co')
            ax.plot(avisolon[0][i+1],avisolat[0][i+1],'co')            
            ax.plot(eddylon[i_eddy], eddylat[i_eddy],'r*')
            plt.xlabel('Longitude/°E')
            plt.ylabel('Latitude/°N')
            
            #ax.etopo()
            #plt.legend(labels = 'adt:%sm\nsh:%sm\nsla:%sm\ndist:%skm\ntime_d:%sh' %(adt_d[0][i_sh],sh_d[0][i_sh],sla_d[0][i_sh],distance,time_diff))
            #plt.annotate('adt:%sm\nsh:%sm\nsla:%sm\ndist:%skm\ntime_d:%sh' %(adt_d[0][i_sh],sh_d[0][i_sh],sla_d[0][i_sh],distance,time_diff), xy = (0, 1))
            plt.contourf(X, Y, c, 8, alpha = 0.75, cmap = plt.cm.hot)
             
            a=plt.colorbar()
            a.ax.set_ylabel('SLA(m)')

            #plt.text(slalon[slalon_index_min]+0.1,slalat[slalat_index_min]+0.1, 'adt:%sm\nsh:%sm\nsla:%sm\ndist:%skm\ntime_d:%sh' %(adt_d[0][i_sh],sh_d[0][i_sh],sla_d[0][i_sh],distance,time_diff)) 
            #plt.text(slalon[slalon_index_min]+0.1,slalat[slalat_index_max]-1.2, 'sh-adt:%sm\nsh-sla:%sm\nargo-aviso:%skm\nargo-aviso:%skm' %(sh_d[0][i_sh]-adt_d[0][i_sh],sh_d[0][i_sh]-sla_d[0][i_sh],dist_argo_aviso_a,dist_argo_aviso_b))
            #plt.axis('scaled')
            # ax.set_xlim(-4, 4)
            # ax.set_ylim(-4, 4)
            plt.savefig('E:\\Python\\sh-adt-argo\\figure\\%s.png'%i_sh,dpi=500)
            #plt.axis('equal')   #changes limits of x or y axis so that equal increments of x and y have the same length
            
            plt.show()


for i_p in range(len(sh_d[0])):
    plt.plot(i_p,adt_d[0][i_p],'ro')
    plt.plot(i_p,sh_d[0][i_p],'b+')
plt.xlabel('number')
plt.ylabel('height/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\12345.png',dpi=500)
plt.show()    
for i_p in range(len(sh_d[0])):
    plt.plot(i_p,sla_d[0][i_p],'ro')
    plt.plot(i_p,sh_d[0][i_p],'b+')
plt.xlabel('Number')
plt.ylabel('Height/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\123456.png',dpi=500)
plt.show()                  

'''plt.plot(distance_all,sh_adt,'ro')
plt.xlabel(distance)
plt.savefig('E:\\Python\\result\\argo_sh\\%s.png'%i_sh,dpi=500)
plt.plot(time_different_all,sh_adt,'ro')'''
a=[0,1,3,4,5,6,7,8,10,12,13,14,16,17,18,19,20,21,22,23,24,26,29,30,31,32]
#a=[0,1,3,10,12,13,14,16,19,20,21,23,26]
#a=[3,10,12,13,16,19,23]#21,
x=range(len(distance_all))
sh_d_a=[sh_d[0][i] for i in a]
adt_d_a=[adt_d[0][i] for i in a]
sla_d_a=[sla_d[0][i] for i in a]
'''b=[0,1,2,8,9,10,11,12,13,15,16,17,19,21]
x=range(len(distance_all))
sh_d_b=[sh_d_a[0][p] for p in b]
adt_d_b=[adt_d_a[0][p] for p in b]
sla_d_b=[sla_d_a[0][p] for p in b]'''

num=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
plt.plot(num,sh_d_a,'ro')
plt.plot(num,adt_d_a,'y+')
plt.plot(num,sla_d_a,'bs')
plt.xlabel('Number')
plt.ylabel('Different between two points/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\123456789.png',dpi=500)
plt.show()

#plt.plot(time_distance,sh_adt_abs,'bo', label='adt')
#plt.plot(time_distance,sh_sla_abs,'r+', label='sla')
plt.xlabel('distance*time')
plt.ylabel('sh-adt/sh-sla')
plt.legend(loc="upper left")


#'sla_d_a':sla_d_a,'adt_d_a':adt_d_a,'sh_d_a':sh_d_a,

scipy.io.savemat('result_data_test0.150.21228.mat',{'sla_d_a':sla_d_a,'adt_d_a':adt_d_a,'sh_d_a':sh_d_a,'sla_d':sla_d,'adt_d':adt_d,'sh_d':sh_d,'wind_speed':windspeed,'time_different':time_different_all, 'distance_all':distance_all,'sh_adt':sh_adt,'sh_sla':sh_sla,'dist_argo_aviso':dist_all_argo_aviso,'dist_argo_eddy':dist_all_argo_eddy,'timediff_argo_aviso':aatime_different_all})















