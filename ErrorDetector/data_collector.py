import time
import os
import tensorflow.compat.v2 as tf
import numpy as np
import pandas as pd

import platform
import socket, struct


old_ts_length = 150
threshold = 0.4
prior_idx = 1

os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
tf.enable_v2_behavior()

udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
udp_socket.bind(("127.0.0.1", 54320))
count = 0
ncount = 0
data_collect = np.array([])
tdelt = 0
print('WAITING FOR SIMULINK')

for count in range(0, 3801, 1):
    recv_data = udp_socket.recvfrom(128)
    recv_msg = recv_data[0]
    send_addr = recv_data[1]
    recv_msg_decode = struct.unpack("d", recv_msg)[0]
    data_collect = np.append(data_collect, recv_msg_decode)
    if count == 0:
        print('RECEIEVING')
    else:
        print(f'Length of data {len(data_collect)}--> {data_collect}')

udp_socket.close()
DF = pd.DataFrame(data_collect)
DF.to_csv("data/exo_hip_right_" + time.strftime('%Y_%m_%d-%H_%M') + ".csv", index=False)
