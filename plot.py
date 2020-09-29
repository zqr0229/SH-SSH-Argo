
#analysis_argo_result 的下一步，将结果数据进行分析图片绘制


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

time_distance=[]
time_different_all=[]
sh_adt_abs=[]
sh_sla_abs=[]
dist_add=[]
dist_minus=[]
dist_minus_abs=[]
dist_argo_eddy_add=[]
dist_argo_eddy_minus_abs=[]
dist_argo_eddy_minus=[]
wind_speed=[]
wind_add=[]
wind_minus=[]
wind_dis=[]
diff_adt_sla=[]

x = np.linspace(0, 210, 1000)
y=0*x
data = scipy.io.loadmat('E:\\Python\\sh-adt-argo\\0.150.2\\result_data_test0.150.2.mat')

#两个argo的时间差
time_different=data['time_different']
time_different=time_different[0]
#time_different=[-0.13333333333139308, 0.066666666622040793, 0.016666666633682325, 0.066666666622040793, -0.16666666659875773, 0.0, 0.1836111111624632, -0.19999999995343387, 0.033333333354676142, 0.16666666659875773, 0.35027777776122093, -0.033333333354676142, 0.19999999995343387, 0.1499999999650754, 0.1836111111624632, -0.19999999995343387, -0.049999999988358468, -0.38361111111589707, -0.21694444442982785, 0.016666666633682325, -0.16694444444146939, -0.20027777779614553, 0.23361111106351018, 0.049999999988358468, 0.016666666633682325, -0.18361111107515171, -0.21694444442982785, 0.11694444445311092, -0.033333333354676142, 0.13333333333139308, 0.033333333354676142, -0.28333333329646848, -0.033611111110076308]
#两个argo的距离差
distance_all=data['distance_all']#argo两点间距离
distance_all=distance_all[0]
#distance_all=[100.51155734035542, 43.752702059482395, 18.952911677934523, 35.13889949969884, 199.93382838329794, 209.80731366396327, 147.8597601946375, 207.07702706041445, 12.545617419921753, 31.016141220918083, 54.116675336200316, 27.860337187799313, 63.40730613790125, 94.82168919049741, 73.90870424217427, 3.5977523335214867, 32.470995805045575, 70.4116317077227, 120.78086780659893, 21.610640396460912, 28.616102030119958, 21.876967713969634, 101.15267228534591, 96.91277078718312, 100.73268279285631, 22.907966289581722, 0.4858739838736094, 8.745805350101735, 22.553648743932754, 154.99351909388753, 43.44859954889374, 190.68442847189866, 142.5973397381354]
#SH-ADT
wind_speed=data['wind_speed']
sh_adt=data['sh_adt']
sh_adt=sh_adt[0]
#sh_adt=[-0.047718224813144472, -0.038709939187858922, -0.018619188362541728, -0.015582301969484735, -0.043423465950667595, -0.029806168675909905, 0.0039213658046144406, -0.042185393941602811, 0.00013695955586268127, 0.013617297274757689, 0.047344831755282035, 0.0012380720090647834, 0.0090082856252855503, 0.029099036450602744, 0.033727534480524346, -0.012379225265692906, 0.020090750825317194, -0.046106759746217252, -0.02786829365293686, 0.0047205761319131234, -0.029357416519905977, 0.004634422655360515, 0.032588869784849983, -0.001489122866969117, 0.032502716308297375, -0.0340779926518191, -8.615347655260841e-05, -0.007269664830277911, 0.033991839175266492, -0.091341423493192631, 0.016753683012947773, 0.076013272520762287, -0.082882200908819481]
#SH-SLA
sh_sla=data['sh_sla']
sh_sla=sh_sla[0]
#sh_sla=[-0.039718224813144631, -0.036709939187859003, -0.018619188362541728, -0.016582301969484639, -0.052423465950667436, -0.038806168675909747, -0.0010786341953854806, -0.051185393941602653, -0.00086304044413731962, 0.013617297274757689, 0.051344831755281956, 0.0012380720090647834, 0.0030082856252856283, 0.021099036450602904, 0.037727534480524266, -0.012379225265692906, 0.018090750825317276, -0.050106759746217172, -0.028868293652936722, 0.0057205761319133047, -0.028357416519905795, 0.0056344226553606963, 0.034588869784850027, 0.00051087713303092641, 0.034502716308297418, -0.0340779926518191, -8.615347655260841e-05, -0.007269664830277911, 0.033991839175266492, -0.077341423493192507, 0.033753683012947885, 0.053013272520762184, -0.070882200908819332]
#argo和sla的距离差
dist_argo_aviso=data['dist_argo_aviso']
dist_argo_aviso=dist_argo_aviso[0]
#dist_argo_aviso=[17.77546443085046, 14.610319431672318, 17.77546443085046, 12.821377053535237, 17.77546443085046, 2.3140974607823335, 18.50488422867081, 14.916769761251157, 4.357287785543216, 10.714690027351987, 4.357287785543216, 20.303435390772368, 4.357287785543216, 21.69689076382494, 4.357287785543216, 17.19368019710252, 11.523521394637639, 11.771025390552685, 10.714690027351987, 20.303435390772368, 10.714690027351987, 21.69689076382494, 10.714690027351987, 17.19368019710252, 14.610319431672318, 12.821377053535237, 14.610319431672318, 2.3140974607823335, 20.303435390772368, 21.69689076382494, 20.303435390772368, 17.19368019710252, 12.821377053535237, 2.3140974607823335, 21.69689076382494, 17.19368019710252, 10.412277780244782, 17.442907221048706, 10.412277780244782, 14.43906119663959, 10.412277780244782, 9.477791600014971, 10.412277780244782, 14.017465422720996, 17.442907221048706, 14.43906119663959, 17.442907221048706, 9.477791600014971, 17.442907221048706, 14.017465422720996, 14.43906119663959, 9.477791600014971, 14.43906119663959, 14.017465422720996, 13.228529974148264, 12.881335812023876, 9.477791600014971, 14.017465422720996, 16.324098707628725, 9.249822304901517, 14.08950750970602, 9.258606544311842, 19.82085717199964, 11.788574770051737, 23.184768921287656, 9.987611392360996]
#argo和涡心的距离差
dist_argo_eddy=data['dist_argo_eddy']
dist_argo_eddy=dist_argo_eddy[0]
#dist_argo_eddy=[42.817508257378506, 111.78890780017416, 42.817508257378506, 49.118632749167496, 42.817508257378506, 27.190305919352063, 50.458073163978774, 27.897096249069573, 142.12892655117932, 77.70530405782574, 142.12892655117932, 73.45896747112135, 142.12892655117932, 57.176013079416755, 142.12892655117932, 71.66754727451543, 229.24189927052942, 229.86126831219534, 77.70530405782574, 73.45896747112135, 77.70530405782574, 57.176013079416755, 77.70530405782574, 71.66754727451543, 111.78890780017416, 49.118632749167496, 111.78890780017416, 27.190305919352063, 73.45896747112135, 57.176013079416755, 73.45896747112135, 71.66754727451543, 49.118632749167496, 27.190305919352063, 57.176013079416755, 71.66754727451543, 100.45965261797444, 53.39288175865022, 100.45965261797444, 87.61384690517714, 100.45965261797444, 71.9234922328694, 100.45965261797444, 87.12819480883962, 53.39288175865022, 87.61384690517714, 53.39288175865022, 71.9234922328694, 53.39288175865022, 87.12819480883962, 87.61384690517714, 71.9234922328694, 87.61384690517714, 87.12819480883962, 169.36954713226422, 177.11779673903365, 71.9234922328694, 87.12819480883962, 81.14516359704076, 235.45649116746662, 199.47414918436687, 168.91735281865175, 97.92163785892618, 242.82964390181948, 415.3340019103139, 369.36973918384075]
for i in range(len(distance_all)):
    time_distance.append(abs(distance_all[i])*abs(time_different[i]))
    time_different_all.append(abs(time_different[i]))#两个argo的绝对时间差
    sh_adt_abs.append(abs(sh_adt[i]))#SH-ADT的绝对值
    sh_sla_abs.append(abs(sh_sla[i]))#SH-SLA的绝对值
    dist_add.append(abs(dist_argo_aviso[2*i])+abs(dist_argo_aviso[2*i+1]))#两个argo和sla的距离差之和
    dist_minus.append(abs(dist_argo_aviso[2*i])-abs(dist_argo_aviso[2*i+1]))#两个argo和sla的距离差之差
    dist_minus_abs.append(abs(abs(dist_argo_aviso[2*i])-abs(dist_argo_aviso[2*i+1])))#两个argo和sla的距离差之差的绝对值
    dist_argo_eddy_add.append(abs(dist_argo_eddy[2*i])+abs(dist_argo_eddy[2*i+1]))#两个argo和涡心的距离差之和
    dist_argo_eddy_minus.append(abs(dist_argo_eddy[2*i])-abs(dist_argo_eddy[2*i+1]))#两个argo和涡心的距离差之差
    dist_argo_eddy_minus_abs.append(abs(abs(dist_argo_eddy[2*i])-abs(dist_argo_eddy[2*i+1])))#两个argo和涡心的距离差之差的绝对值
    wind_minus.append(abs(abs(wind_speed[0][2*i])-abs(wind_speed[0][2*i+1])))
    wind_add.append(abs(wind_speed[0][2*i])+abs(wind_speed[0][2*i+1]))
    wind_dis.append(abs(wind_speed[0][2*i])+abs(wind_speed[0][2*i+1])*abs(distance_all[i]))

#--------去除sla只有一个点的情况----
a=[0,1,3,4,5,6,7,8,10,12,13,14,16,17,18,19,20,21,22,23,24,26,29,30,31,32]
#b=[1,3,14,16,17]
#a=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]
x=range(len(distance_all))
distance_all_a=[distance_all[i] for i in a]
sh_adt_abs_a=[sh_adt_abs[i] for i in a]
sh_sla_abs_a=[sh_sla_abs[i] for i in a]
#sh_adt_abs_b=[sh_adt_abs[i] for i in b]
#sh_sla_abs_b=[sh_sla_abs[i] for i in b]
sh_adt_a=[sh_adt[i] for i in a]
sh_sla_a=[sh_sla[i] for i in a]
dist_add_a=[dist_add[i] for i in a]
dist_minus_abs_a=[dist_minus_abs[i] for i in a]
dist_argo_eddy_add_a=[dist_argo_eddy_add[i] for i in a]
#dist_argo_eddy_add_b=[dist_argo_eddy_add[i] for i in b]
wind_minus_a=[wind_minus[i] for i in a]
#-----------------------------------adt----abs-----------    

'''plt.plot(distance_all,sh_adt_abs,'bo', label='adt',markersize=3)
plt.plot(distance_all,sh_sla_abs,'r+', label='sla',markersize=3)
plt.legend(loc="upper left")
plt.xlabel('distance between two argo/km')
plt.ylabel('abs of sh-adt/sh-sla/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adtsla_distance.png',dpi=500)
plt.show()'''


for i_a in range(len(distance_all_a)):
    if i_a==6 or i_a==24:
        plt.text(distance_all_a[i_a]+2,sh_adt_abs_a[i_a]+0.1,'%s'%i_a)

plt.plot(distance_all_a,sh_adt_abs_a,'b*',markersize=5,label='adt')

plt.plot(distance_all_a,sh_sla_abs_a,'r+',markersize=5, label='sla')
   # plt.text(distance_all_a[i_a]+0.1,sh_sla_abs_a[i_a]+0.1,'%s'%i_a)
plt.legend(loc="upper left")
x = np.array(distance_all_a)
y = np.array(sh_adt_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b',markersize=1) 
x = np.array(distance_all_a)
y = np.array(sh_sla_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'r',markersize=1) 
#plt.plot(x,y)
plt.xlabel('Distance between two Argos/km')
plt.ylabel('Height difference of SD-AD in two points/cm')
plt.ylim(0,25)
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adtsla01.png',dpi=500)
plt.show()

'''for i_a in range(len(distance_all_a)):
    plt.plot(distance_all_a[i_a],sh_sla_abs_a[i_a],'ro',markersize=3)
    plt.text(distance_all_a[i_a]+0.1,sh_sla_abs_a[i_a]+0.1,'%s'%i_a)
x = np.array(distance_all_a)
y = np.array(sh_sla_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b') 
#plt.plot(x,y)
plt.xlabel('distance between two argo/km')
plt.ylabel('absolute different between sh and sla/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\sla01.png',dpi=500)
plt.show()'''

plt.plot(distance_all_a,sh_adt_a,'b*',label='adt')
plt.plot(distance_all_a,sh_sla_a,'r+',label='sla')
plt.plot(x,0*x,linewidth=1)
for i_asd in range(len(distance_all_a)):
    diff_adt_sla.append(abs(sh_adt_a[i_asd]-sh_sla_a[i_asd]))
plt.plot(distance_all_a,diff_adt_sla,'c.',label='adt-sla')
plt.legend(loc="upper left")
plt.xlabel('Distance between two Argos/km')
plt.ylabel('Height difference of SD-AD in two points/cm')
plt.ylim(-25,25)
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adtsla11.png',dpi=500)
plt.show()

'''plt.plot(distance_all_a,sh_sla_a,'ro')
plt.plot(x,0*x)
plt.xlabel('distance between two argo/km')
plt.ylabel('different between sh and sla/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\sla11.png',dpi=500)
plt.show()'''



'''plt.plot(dist_add_a,sh_adt_abs_a,'ro', markersize=3)
plt.xlabel('Add the two distance between Argo and  alongtrack point/km')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adt21.png',dpi=500)
plt.show()

plt.plot(dist_add_a,sh_sla_abs_a,'ro', markersize=3)
plt.xlabel('Add the two distance between Argo and  alongtrack point/km')
plt.ylabel('abs of sh-sla/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\sla21.png',dpi=500)
plt.show()'''

for i_a in range(len(distance_all_a)):
    if i_a==13:
        plt.text(dist_minus_abs_a[i_a]+0.07,sh_adt_abs_a[i_a]+0.1,'%s'%i_a)
   

plt.plot(dist_minus_abs_a,sh_adt_abs_a,'b*',markersize=5,label='adt')

plt.plot(dist_minus_abs_a,sh_sla_abs_a,'r+',markersize=5, label='sla')
   # plt.text(distance_all_a[i_a]+0.1,sh_sla_abs_a[i_a]+0.1,'%s'%i_a)
plt.legend(loc="upper left")
x = np.array(dist_minus_abs_a)
y = np.array(sh_adt_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b',linewidth=2) 
x = np.array(dist_minus_abs_a)
y = np.array(sh_sla_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'r',linewidth=1) 
#plt.plot(x,y)
plt.xlabel('Distance between Argo and satellite point/km')
plt.ylabel('Height difference of SD-AD in two points/cm')
plt.ylim(0,25)
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adtsla31.png',dpi=500)
plt.show()


'''for i_a in range(len(distance_all_a)):
    plt.plot(dist_minus_abs_a[i_a],sh_sla_abs_a[i_a],'ro', markersize=3)
    plt.text(dist_minus_abs_a[i_a]+0.1,sh_sla_abs_a[i_a]+0.1,'%s'%i_a)
x = np.array(dist_minus_abs_a)
y = np.array(sh_sla_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b') 
plt.xlabel('abs of subtracting  the two distance between Argo and  alongtrack point/km')
plt.ylabel('abs of sh-sla/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\sla31.png',dpi=500)
plt.show()

x = np.array(dist_minus_abs_a)
y = np.array(sh_adt_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b') 
for i_a in range(len(distance_all_a)):
    plt.plot(dist_minus_abs_a[i_a],sh_adt_abs_a[i_a],'ro', markersize=3)
    plt.text(dist_minus_abs_a[i_a]+0.1,sh_adt_abs_a[i_a]+0.1,'%s'%i_a)
plt.xlabel('abs of subtracting  the two distance between Argo and  alongtrack point/km')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adt31.png',dpi=500)
plt.show()'''


for i_a in range(len(distance_all_a)):
    if i_a==5 or i_a==6 or i_a==13 or  i_a==25:
        plt.text(dist_argo_eddy_add_a[i_a]+1.5,sh_adt_abs_a[i_a]+0.1,'%s'%i_a)



plt.plot(dist_argo_eddy_add_a,sh_adt_abs_a,'b*',markersize=5,label='adt')

plt.plot(dist_argo_eddy_add_a,sh_sla_abs_a,'r+',markersize=5, label='sla')
   # plt.text(distance_all_a[i_a]+0.1,sh_sla_abs_a[i_a]+0.1,'%s'%i_a)
plt.legend(loc="upper left")
x = np.array(dist_argo_eddy_add_a)
y = np.array(sh_adt_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b',linewidth=1) 
x = np.array(dist_argo_eddy_add_a)
y = np.array(sh_sla_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'r',linewidth=1) 
#plt.plot(x,y)
plt.xlabel('Distance between Argo and eddy center point/km')
plt.ylabel('Height difference of SD-AD in two points/cm')
plt.ylim(0,25)
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adtsla41.png',dpi=500)
plt.show()

'''x = np.array(dist_argo_eddy_add_a)
y = np.array(sh_adt_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b') 
for i_a in range(len(distance_all_a)):
    plt.plot(dist_argo_eddy_add_a[i_a],sh_adt_abs_a[i_a],'ro',markersize=3)
    plt.text(dist_argo_eddy_add_a[i_a]+0.1,sh_adt_abs_a[i_a]+0.1,'%s'%i_a)#,fontsize=5)
plt.xlabel('Add the two distance between Argo and  eddy center point/km')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adt41.png',dpi=500)
plt.show()

x = np.array(dist_argo_eddy_add_a)
y = np.array(sh_sla_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b') 
for i_a in range(len(distance_all_a)):
    plt.plot(dist_argo_eddy_add_a[i_a],sh_sla_abs_a[i_a],'ro',markersize=3)
    plt.text(dist_argo_eddy_add_a[i_a]+0.1,sh_sla_abs_a[i_a]+0.1,'%s'%i_a)
plt.xlabel('Add the two distance between Argo and  eddy center point/km')
plt.ylabel('abs of sh-sla/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\sla41.png',dpi=500)
plt.show()'''


'''x = np.array(dist_argo_eddy_add_b)
y = np.array(sh_adt_abs_b)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b') 
for i_a in range(len(dist_argo_eddy_add_b)):
    plt.plot(dist_argo_eddy_add_b[i_a],sh_adt_abs_b[i_a],'ro',markersize=3)
    plt.text(dist_argo_eddy_add_b[i_a]+0.1,sh_adt_abs_b[i_a]+0.1,'%s'%i_a)#,fontsize=5)
plt.xlabel('Add the two distance between Argo and  eddy center point/km')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adt41b.png',dpi=500)
plt.show()

x = np.array(dist_argo_eddy_add_b)
y = np.array(sh_sla_abs_b)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b') 
for i_a in range(len(dist_argo_eddy_add_b)):
    plt.plot(dist_argo_eddy_add_b[i_a],sh_sla_abs_b[i_a],'ro',markersize=3)
    plt.text(dist_argo_eddy_add_b[i_a]+0.1,sh_sla_abs_b[i_a]+0.1,'%s'%i_a)
plt.xlabel('Add the two distance between Argo and  eddy center point/km')
plt.ylabel('abs of sh-sla/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\sla41b.png',dpi=500)
plt.show()'''
#for i_a in range(len(distance_all_a)):
    #plt.text(wind_minus_a[i_a]+0.01,sh_adt_abs_a[i_a]+0.1,'%s'%i_a)

plt.plot(wind_minus_a,sh_adt_abs_a,'b*',markersize=5,label='adt')

plt.plot(wind_minus_a,sh_sla_abs_a,'r+',markersize=5, label='sla')
   # plt.text(distance_all_a[i_a]+0.1,sh_sla_abs_a[i_a]+0.1,'%s'%i_a)
plt.legend(loc="upper left")
x = np.array(wind_minus_a)
y = np.array(sh_adt_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b',linewidth=1) 
x = np.array(wind_minus_a)
y = np.array(sh_sla_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'r',linewidth=1) 
#plt.plot(x,y)
plt.xlabel('Wind speeds/ms^-1')
plt.ylabel('Height difference of SD-AD in two points/cm')
plt.ylim(0,25)
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adtsla51.png',dpi=500)
plt.show()

'''
x = np.array(wind_minus_a)
y = np.array(sh_adt_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b') 
for i_a in range(len(distance_all_a)):
    plt.plot(wind_minus_a[i_a],sh_adt_abs_a[i_a],'ro',markersize=3)
    plt.text(wind_minus_a[i_a]+0.01,sh_adt_abs_a[i_a]+0.01,'%s'%i_a)
#plt.plot(wind_minus,sh_adt_abs,'ro',markersize=3)
plt.xlabel('minus two wind speed/m/s')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adt51.png',dpi=500)
plt.show()

x = np.array(wind_minus_a)
y = np.array(sh_sla_abs_a)
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m,c)
plt.plot(x, m*x + c, 'b') 
for i_a in range(len(distance_all_a)):
    plt.plot(wind_minus_a[i_a],sh_sla_abs_a[i_a],'ro',markersize=3)
    plt.text(wind_minus_a[i_a]+0.01,sh_sla_abs_a[i_a]+0.01,'%s'%i_a)
#plt.plot(wind_minus,sh_adt_abs,'ro',markersize=3)
plt.xlabel('minus two wind speed/m/s')
plt.ylabel('abs of sh-sla/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\sla51.png',dpi=500)
plt.show()'''

'''
plt.plot(wind_dis,sh_adt_abs,'ro',markersize=3)
plt.xlabel('wind*dis/m/s')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_wind_dis.png',dpi=500)
plt.show()

plt.plot(wind_add,sh_adt_abs,'ro',markersize=3)
plt.xlabel('add two wind speed/m/s')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_wind.png',dpi=500)
plt.show()
plt.plot(wind_minus,sh_adt_abs,'ro',markersize=3)
plt.xlabel('minus two wind speed/m/s')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\121212.png',dpi=500)
plt.show()

plt.plot(dist_add,sh_adt_abs,'ro',markersize=3)
plt.xlabel('Add the two distance between Argo and  alongtrack point/km')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_add.png',dpi=500)
plt.show()
plt.plot(dist_minus,sh_adt_abs,'ro',markersize=3)
plt.xlabel('Subtracting  the two distance between Argo and  alongtrack point/km')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_minus.png',dpi=500)
plt.show()
plt.plot(dist_minus_abs,sh_adt_abs,'ro',markersize=3)
plt.xlabel('abs of Subtracting  the two distance between Argo and  alongtrack point/km')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_minus_abs.png',dpi=500)
plt.show()
plt.plot(dist_argo_eddy_add,sh_adt_abs,'ro',markersize=3)
plt.xlabel('Add the two distance between Argo and  eddy center point/km')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_argo_eddy_add.png',dpi=500)
plt.show()
plt.plot(dist_argo_eddy_minus,sh_adt_abs,'ro',markersize=3)
plt.xlabel('Subtracting the two distance between Argo and  eddy center point/km')
plt.ylabel('abs of sh-adt/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_argo_eddy_minus_abs.png',dpi=500)
plt.show()


#-----------------------------------adt、sla-------abs--------
plt.plot(distance_all,sh_adt_abs,'bo', label='adt',markersize=3)
plt.plot(distance_all,sh_sla_abs,'r+', label='sla',markersize=3)
plt.legend(loc="upper left")
plt.xlabel('distance between two argo/km')
plt.ylabel('abs of sh-adt/sh-sla/cm')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adtsla_distance.png',dpi=500)
plt.show()'''
'''plt.plot(time_different_all,sh_adt_abs,'bo', label='adt')
plt.plot(time_different_all,sh_sla_abs,'r+', label='sla')
plt.xlabel('time')
plt.ylabel('sh-adt/sh-sla')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adtsla_time.png',dpi=500)
plt.show()

plt.plot(time_distance,sh_adt_abs,'bo', label='adt')
plt.plot(time_distance,sh_sla_abs,'r+', label='sla')
plt.xlabel('distance*time')
plt.ylabel('sh-adt/sh-sla')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adtsla_time_distance.png',dpi=500)
plt.show()'''
'''plt.plot(dist_add,sh_adt_abs,'bo', label='adt',markersize=3)
plt.plot(dist_add,sh_sla_abs,'r+', label='sla',markersize=3)
plt.xlabel('Add the two distance between Argo and  alongtrack point/km')
plt.ylabel('abs of sh-adt/cm')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_add.png',dpi=500)
plt.show()
plt.plot(dist_minus,sh_sla_abs,'r+', label='sla',markersize=3)
plt.plot(dist_minus,sh_adt_abs,'bo', label='adt',markersize=3)
plt.xlabel('Subtracting  the two distance between Argo and  alongtrack point/km')
plt.ylabel('abs of sh-adt/cm')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_minus.png',dpi=500)
plt.show()

plt.plot(dist_minus_abs,sh_adt_abs,'bo', label='adt',markersize=3)
plt.plot(dist_minus_abs,sh_sla_abs,'r+', label='sla',markersize=3)
plt.xlabel('abs of Subtracting  the two distance between Argo and  alongtrack point/km')
plt.ylabel('abs of sh-adt/sh-sla/cm')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adtsla_dist_minus.png',dpi=500)
plt.show()

plt.plot(dist_argo_eddy_add,sh_sla_abs,'r+', label='sla',markersize=3)
plt.plot(dist_argo_eddy_add,sh_adt_abs,'bo', label='adt',markersize=3)
plt.xlabel('Add the two distance between Argo and  eddy center point')
plt.ylabel('abs of sh-adt/cm')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_argo_eddy_add.png',dpi=500)
plt.show()

plt.plot(dist_argo_eddy_minus,sh_sla_abs,'r+', label='sla',markersize=3)
plt.plot(dist_argo_eddy_minus,sh_adt_abs,'bo', label='adt',markersize=3)
plt.xlabel('Subtracting the two distance between Argo and  eddy center point/km')
plt.ylabel('abs of sh-adt/cm')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_argo_eddy_minus_abs.png',dpi=500)
plt.show()

plt.plot(wind_minus,sh_sla_abs,'r+', label='sla',markersize=3)
plt.plot(wind_minus,sh_adt_abs,'bo', label='adt',markersize=3)
plt.xlabel('minus two wind speed/m/s')
plt.ylabel('abs of sh-adt/cm')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_wind_minus_abs.png',dpi=500)
plt.show()

plt.plot(wind_add,sh_sla_abs,'r+', label='sla',markersize=3)
plt.plot(wind_add,sh_adt_abs,'bo', label='adt',markersize=3)
plt.xlabel('add two wind speed/m/s')
plt.ylabel('abs of sh-adt/cm')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_wind_minus_abs.png',dpi=500)
plt.show()

plt.plot(wind_dis,sh_sla_abs,'r+', label='sla',markersize=3)
plt.plot(wind_dis,sh_adt_abs,'bo', label='adt',markersize=3)
plt.xlabel('wind*dis')
plt.ylabel('abs of sh-adt/cm')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_adt_dist_wind_dis_abs.png',dpi=500)
plt.show()

'''
#-----------------------------------adt------
'''plt.plot(distance_all,sh_adt,'ro')
#plt.plot(x,y)
plt.xlabel('distance')
plt.ylabel('sh-adt')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adt_distance.png',dpi=500)
plt.show()
'''
'''
#----------------------------------、sla-------------------------
plt.plot(distance_all,sh_sla,'ro')
#plt.plot(x,y)
plt.xlabel('distance')
plt.ylabel('sh_sla')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\sla_distance.png',dpi=500)
plt.show()
'''



#-----------------------------------adt、sla----   -------
'''
plt.plot(distance_all,sh_adt,'bo', label='adt')
plt.plot(distance_all,sh_sla,'r+', label='sla')
#plt.plot(x,y)
plt.xlabel('distance')
plt.ylabel('sh-adt/sh-sla')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adtsla_distance.png',dpi=500)
plt.show()
'''

'''plt.plot(time_different_all,sh_adt,'bo', label='adt')
plt.plot(time_different_all,sh_sla,'r+', label='sla')
plt.xlabel('time')
plt.ylabel('sh-adt/sh-sla')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adtsla_time.png',dpi=500)
plt.show()

plt.plot(time_distance,sh_adt,'bo', label='adt')
plt.plot(time_distance,sh_sla,'r+', label='sla')
plt.xlabel('distance*time')
plt.ylabel('sh-adt/sh-sla')
plt.legend(loc="upper left")
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\adtsla_time_distance.png',dpi=500)
plt.show()
'''
#-----------------------------------sla------abs---------

'''plt.plot(time_different_all,sh_sla_abs,'ro')
plt.xlabel('time')
plt.ylabel('sh-sla')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_sla_time.png',dpi=500)
plt.show()
plt.xlabel('distance*time')
plt.ylabel('sh-sla')
plt.plot(time_distance,sh_sla_abs,'ro')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_sla_time_distance.png',dpi=500)
plt.show()'''
'''plt.plot(dist_add,sh_sla_abs,'ro',markersize=3)
plt.xlabel('distance between argo and alongtrack')
plt.ylabel('sh-sla')
plt.savefig('E:\\Python\\sh-adt-argo\\figure\\abs_sla_dist_add.png',dpi=500)
plt.show()'''


scipy.io.savemat('plot_data0.150.2.mat',{'distance_argo':distance_all,'time_different':time_different_all, 'distance_all':distance_all,'sh_adt':sh_adt,'sh_sla':sh_sla})
