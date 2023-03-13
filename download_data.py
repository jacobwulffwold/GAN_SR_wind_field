
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from datetime import timedelta , date
import datetime
from urllib import request
import os
import pickle
import glob
import netCDF4 
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import num2date 
import matplotlib.animation as animation
import matplotlib
import cftime
def check (url):
    try:
        u = request.urlopen(url)
        u.close()
        return True
    except:
        return False
            
def download_Bessaker_data(start_date, end_date,destination_folder):
    start_date = start_date  
    end_date   = end_date
    delta = end_date - start_date
    home_dir = 'http://thredds.met.no/thredds/fileServer/opwind/'
    data_code = 'simra_BESSAKER_'
    sim_times = ['T00Z.nc','T12Z.nc'] 
    destination_folder = './'
    no_data_points = (delta.days+1)*2
    counter = 0
    for i in range(delta.days+1):
        for sim_time in sim_times:
            temp=start_date + timedelta(days=i)
            temp_date = datetime.datetime.strptime(str(temp),"%Y-%m-%d")
            filename = data_code+((str(temp)).replace("-",""))
            filename=destination_folder+filename+sim_time
            URL=home_dir+str(temp_date.year)+'/'+str(temp_date.month).zfill(2)+'/'+str(temp_date.day).zfill(2)+'/'+filename          
            if os.path.isfile(filename) == True:
                counter = counter + 1
                print("Number of files downloaded ",counter, '/',no_data_points)
            elif os.path.isfile(filename) != True:
                print("Attempting to download file ", filename)
                try:
                    if check(URL):
                        request.urlretrieve(URL,'./'+filename)
                        print("Downloading file" ,filename)
                        counter = counter + 1
                        print("Number of files downloaded ",counter, '/',no_data_points)
                except TypeError as e:
                    print('Error downlowing file {}'.format(e))
                    print('Continuing')

def combine_files(data_code,start_date,end_date,outfilename):
    start_date = start_date  
    end_date   = end_date
    delta = end_date - start_date
    sim_times = ['T00Z.nc','T12Z.nc']
    nc_dir= list()
    for i in range(delta.days+1):
        for sim_time in sim_times:
            temp=start_date + timedelta(days=i)
            filename = data_code+((str(temp)).replace("-",""))
            filename = filename+sim_time
            nc_dir.append(filename)      
    #list_of_paths = glob.glob(nc_dir, recursive=True)
    nc_fid = netCDF4.MFDataset(nc_dir)
    latitude  = nc_fid['longitude'][:]
    longitude = nc_fid['latitude'][:]
    x = nc_fid['x'][:]
    y = nc_fid['y'][:]
    z = nc_fid['geopotential_height_ml'][1,:,:,:]
    terrain = nc_fid['surface_altitude'][:]
    theta = nc_fid['air_potential_temperature_ml'][:]
    u = nc_fid['x_wind_ml'][:]
    v = nc_fid['y_wind_ml'][:]
    w = nc_fid['upward_air_velocity_ml'][:]
    u10 = nc_fid['x_wind_10m'][:]
    v10 = nc_fid['y_wind_10m'][:]
    tke = nc_fid['turbulence_index_ml'][:]
    td  = nc_fid['turbulence_dissipation_ml'][:]
    outfile = open(outfilename,'wb')
    pickle.dump([latitude, longitude,x,y,z,u,v,w,theta,tke,td,u10,v10,terrain],outfile)
    outfile.close()

def extract_XY_plane(data_code,start_date,end_date,iz):
    data_code = data_code
    start_date = start_date  
    end_date   = end_date
    delta = end_date - start_date
    sim_times = ['T00Z.nc','T12Z.nc']
    index=0
    for i in range(delta.days+1):
        for sim_time in sim_times:
            temp=start_date + timedelta(days=i)
            filename = data_code+((str(temp)).replace("-",""))
            filename = filename+sim_time
            if index == 0:
                nc_fid = Dataset(filename,mode='r')
                time = nc_fid['time'][:]
                latitude  = nc_fid['longitude'][:]
                longitude = nc_fid['latitude'][:]
                x = nc_fid['x'][:]
                y = nc_fid['y'][:]
                z = nc_fid['geopotential_height_ml'][:][1,iz,:,:]
                terrain = nc_fid['surface_altitude'][:]
                theta   = nc_fid['air_potential_temperature_ml'][:][:,iz,:,:]
                u       = nc_fid['x_wind_ml'][:][:,iz,:,:]
                v       = nc_fid['y_wind_ml'][:][:,iz,:,:]
                w       = nc_fid['upward_air_velocity_ml'][:][:,iz,:,:]
                #u10 = nc_fid['x_wind_10m'][:]
                #v10 = nc_fid['y_wind_10m'][:]
                tke = nc_fid['turbulence_index_ml'][:][:,iz,:,:]
                td  = nc_fid['turbulence_dissipation_ml'][:][:,iz,:,:]
                index = index+1
                nc_fid.close()
            else:
                nc_fid = Dataset(filename,mode='r')
                time   = np.ma.append(time,nc_fid['time'][:][1:13],axis=0)
                u      = np.ma.append(u,nc_fid['x_wind_ml'][1:13,iz,:,:],axis=0)
                v      = np.ma.append(v,nc_fid['y_wind_ml'][:][1:13,iz,:,:],axis=0)
                w      = np.ma.append(w,nc_fid['upward_air_velocity_ml'][:][1:13,iz,:,:],axis=0)
                theta  = np.ma.append(theta,nc_fid['air_potential_temperature_ml'][:][1:13,iz,:,:],axis=0)
                tke    = np.ma.append(tke,nc_fid['turbulence_index_ml'][:][1:13,iz,:,:],axis=0)
                td     = np.ma.append(td,nc_fid['turbulence_dissipation_ml'][:][:][1:13,iz,:,:],axis=0)
                nc_fid.close()
    return time, latitude, longitude, terrain, x,y,z,u,v,w,theta,tke,td 
    
data_code = 'simra_BESSAKER_'
start_date = date(2018, 4, 1)  #1,2
end_date   = date(2018, 4, 3) # 
#download_Bessaker_data(start_date,end_date,'./')
#combine_files(data_code,start_date,end_date,outfilename='temp.pkl')
ix=75
iy=75
iz=20 #the higher the value the closest the layer is to the ground
time, latitude, longitude, terrain, x,y,z,u,v,w,theta,tke,td = extract_XY_plane(data_code,start_date,end_date,iz)

plt.plot(u[:,20,10])
plt.show()

X, Y = np.meshgrid(x, y)
plt.contour(z)
plt.contourf(u[1,:,:],cmap='hsv')
plt.quiver(u[10,:,:],v[10,:,:],scale=200)
plt.show()
print(np.max(u),np.max(v),np.max(w),np.max(theta))


'''
delta_x = (X[135, 134] - X[0, 0])*10**5
delta_y = (Y[135, 134] - Y[0, 0])*10**5
X_res = delta_x/136
Y_res = delta_y/135
'''
# saving u, v, w 
with open('2018_apr.pickle', 'wb') as f:
    pickle.dump([u, v, w], f)
 


'''
with open('2018_january.pickle', 'rb') as f:
   u_jan, v_jan, w_jan = pickle.load(f)
with open('2018_february.pickle', 'rb') as f:
   u_feb, v_feb, w_feb = pickle.load(f)
with open('2018_march.pickle', 'rb') as f:
   u_mar, v_mar, w_mar = pickle.load(f)
with open('2018_apr.pickle', 'rb') as f:
   u_apr, v_apr, w_apr = pickle.load(f)

u = np.concatenate((u_jan,u_feb,u_mar, u_apr), axis = 0)
v = np.concatenate((v_jan,v_feb,v_mar, v_apr), axis = 0)
w = np.concatenate((w_jan,w_feb,w_mar, w_apr), axis = 0)
with open('download_data.pickle', 'wb') as f:
    pickle.dump([u, v, w], f)
'''



 