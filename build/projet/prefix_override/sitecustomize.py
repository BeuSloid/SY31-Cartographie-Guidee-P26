import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/chttioui/sy31_ws/src/SY31-Cartographie-Guidee-P26/install/projet'
