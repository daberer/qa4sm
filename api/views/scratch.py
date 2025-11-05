from qa4sm_reader.img import QA4SMImg
from qa4sm_reader.plotter import QA4SMPlotter
import qa4sm_reader.globals
import os
import tempfile
import qa4sm_reader.plot_all as pa


testfile_path = r'/home/daberer/Downloads/zeyang.nc'
testfile_path = r'/home/daberer/Downloads/0-ISMN.soil_moisture_with_1-AMSR2_0.25_d_201207_202412_X.sm_gapfilled.nc'
testfile_path = '/home/daberer/local_repos/qa4sm_interactive_maps/qa4sm/output/6605acd3-2031-411a-8a22-d0095c68ac68/0-C3S_combined.sm_with_1-SMOS_L3.Soil_Moisture.nc'
#testfile_path = r'/home/daberer/local_repos/qa4sm_stability/qa4sm/output/9f5dd0d7-48a0-400d-a98a-1326d9c9cbcc/0-SMOS_L3.Soil_Moisture_with_1-ERA5_LAND.swvl1.nc'
#testfile_path = r'/home/daberer/local_repos/qa4sm_stability/qa4sm/output/d2b24742-466f-44c0-885f-ee63c46d0d62/0-ISMN.soil_moisture_with_1-ESA_CCI_SM_combined.sm_with_2-ESA_CCI_SM_passive.sm.nc'
#testfile_path = r'/home/daberer/local_repos/qa4sm_stability/qa4sm/output/a427429f-dd4c-4abc-b75d-2c2c3c0c7fb0/0-ERA5_LAND.swvl1_with_1-SMOS_L3.Soil_Moisture.nc'
#testfile_path = r'/home/daberer/Downloads/0-ASCAT.sm_with_1-GLDAS.SoilMoi100_200cm_inst_with_2-C3S_combined.sm.nc'
path = r'/tmp/tmptv4yjpa4_keep'
if not os.path.exists(path):
    os.mkdir(path)
plotdir = r'/tmp/tmptv4yjpa4_keep'

#img = QA4SMImg(testfile_path, ignore_empty=False)
#plotter = QA4SMPlotter(img, plotdir)
#plotter.mapplot_metric('n_obs', out_types='png', save_files=True)

pa.plot_all(filepath=testfile_path, out_dir=plotdir, save_all=True, save_metadata=True, out_type=['png'], engine='netcdf4')

print('script done')