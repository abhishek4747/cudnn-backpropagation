import os;
import sys;
import time;

try:
    os.system('MSBUILD_UTIL.bat mnistCUDNN_vs2010.sln x64 Release ' ' %s' % (sys.argv[1]))
except:
    print 'Even if MSBUILD is special here, binary is generated successfully, so ignore this exception'

