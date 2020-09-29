# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:30:34 2018
找按照num的argo里面的时间和距离范围内的一对对alongtrack

@author: zqr02
"""
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import csv
import scipy.io
import os
import re
import time
from ftplib import FTP
"""

import urllib
import sys
from decimal import *
import numpy as np
import matplotlib 
from mpl_toolkits.basemap import Basemap
import pylab
import pandas as pd
from matplotlib import animation
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
"""
distance=0.25#units is °
timeresolution=0.5#unit is day
start_time=datetime(2014,3,27)
#end_time=datetime(2018,5,17)
#timespace=21#satellite every data time range

ftp=FTP() 
argo_in_alongtrack_id=[]
argo_in_alongtrack_id_ma=[]
argo_in_alongtrack_cycle=[]
argo_in_alongtrack_argotime=[]
argo_in_alongtrack_argolon=[]
argo_in_alongtrack_argolat=[]
argo_in_alongtrack_avisolat=[]
argo_in_alongtrack_avisolon=[]
argo_in_alongtrack_avisotime=[]
argo_in_alongtrack_argosa=[]
argo_in_alongtrack_argotemp=[]
argo_in_alongtrack_argopres=[]
argo_in_alongtrack_avisosla=[]
argo_in_alongtrack_avisoadt=[]
for i in range (2901550,2901551):
    print(i)
    argo_data_url="F:\\data\\ARGO\\for_num\\%s\\%s_prof.nc"%(i,i)
    argo_nc_obj=nc.Dataset(argo_data_url)#读取每个ARGO的数据
    argo_lon=argo_nc_obj.variables['LONGITUDE'][:]
    argo_lat=argo_nc_obj.variables['LATITUDE'][:]
    argo_cycle=argo_nc_obj.variables['CYCLE_NUMBER'][:]
    argo_time=argo_nc_obj.variables['JULD'][:]
    argo_temp=argo_nc_obj.variables['TEMP_ADJUSTED'][:]
    argo_sa=argo_nc_obj.variables['PSAL_ADJUSTED'][:]
    argo_pres=argo_nc_obj.variables['PRES_ADJUSTED'][:]
    print( max(argo_lon),min(argo_lon),max(argo_lat),min(argo_lat))
    for i_argo in range(len(argo_lon)) :
       
        rootdir = 'F:\\data\\AVISO\\SEALEVEL_GLO_PHY_L3_REP_OBSERVATIONS_008_045\\by_time\\2014' 
        for file in os.listdir(rootdir): 
            list = os.listdir(rootdir) #列出文件夹下所有的目录与文件 3 
            for i_av in range(0,len(list)):  #遍历aviso2014文件夹内所有的nc文件#aviso命名方式有问题
            #path=os.path.join(rootdir,list[i_av]) 
            
                file_start_time=start_time+timedelta(i_av-1)
                filetime=file_start_time.strftime("%Y%m%d")
                avisofile="dt_global_.._phy_vfec_l3_%s_.........nc"%filetime
                
                #print (path,aviso_data_url)
                if (re.match(avisofile,file)):
                    aviso_data_url="F:\\data\\AVISO\\SEALEVEL_GLO_PHY_L3_REP_OBSERVATIONS_008_045\\by_time\\2014\\%s"%file
                    print (0)
                    aviso_nc_obj=nc.Dataset(aviso_data_url) #along track data
                    #print(aviso_nc_obj.variables.keys())
                    #lon=aviso_nc_obj.variables['longitude']#along track data introduce
                    #lat=aviso_nc_obj.variables['latitude']#along track data introduce
                    
                    
                    aviso_lon=aviso_nc_obj.variables['longitude'][:]
                    aviso_lat=aviso_nc_obj.variables['latitude'][:]
                    aviso_time=aviso_nc_obj.variables['time'][:]
                    aviso_sla=aviso_nc_obj.variables['sla_filtered'][:]
                    aviso_adt=aviso_nc_obj.variables['adt_filtered'][:]
                    
                    for i_aviso in range (len(aviso_lon)):
                        #print (len(aviso_lon))
                        if  abs(aviso_lon[i_aviso]-argo_lon[i_argo])<distance and abs(aviso_lat[i_aviso]-argo_lat[i_argo])<distance and abs(aviso_time[i_aviso]-argo_time[i_argo])<timeresolution:
                            print (2)
                            argo_in_alongtrack_id_ma.append(i)
                            #argo_in_alongtrack_cycle.append(argo_cycle[i_argo])
                            argo_in_alongtrack_argotime.append(argo_time[i_argo])
                            argo_in_alongtrack_argolon.append(argo_lon[i_argo])
                            argo_in_alongtrack_argolat.append(argo_lat[i_argo])
                            argo_in_alongtrack_avisolat.append(aviso_lat[i_aviso])
                            argo_in_alongtrack_avisolon.append(aviso_lon[i_aviso])
                            argo_in_alongtrack_avisotime.append(aviso_time[i_aviso])
                            argo_in_alongtrack_argosa.append(argo_sa[i_argo])
                            argo_in_alongtrack_argotemp.append(argo_temp[i_argo])
                            argo_in_alongtrack_argopres.append(argo_pres[i_argo])
                            argo_in_alongtrack_avisosla.append(aviso_sla[i_aviso])
                            argo_in_alongtrack_avisoadt.append(aviso_adt[i_aviso])        
scipy.io.savemat('argo_in_alongtrack_test_0.250.5.mat',{'id':argo_in_alongtrack_id_ma, 'argotime': argo_in_alongtrack_argotime, 'argolon':argo_in_alongtrack_argolon,'argolat': argo_in_alongtrack_argolat, 'avisolat': argo_in_alongtrack_avisolat, 'avisotime': argo_in_alongtrack_avisotime, 'avisolon': argo_in_alongtrack_avisolon, 'avisosla': argo_in_alongtrack_avisosla, 'avisoadt': argo_in_alongtrack_avisoadt})               




