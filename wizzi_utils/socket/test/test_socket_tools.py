from wizzi_utils.socket import socket_tools as st
from wizzi_utils.misc import misc_tools as mt
from wizzi_utils.misc.test import test_misc_tools as mtt
from wizzi_utils.json import json_tools as jt
import socket
import os
import threading

SERVER_ADDRESS = ('localhost', 10000)
BUF_LEN = 20
END_MSG = "$#$#"


def connect_to_server():
    connect_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = SERVER_ADDRESS
    print("\tClientOutput:Connecting to server {}".format(server_address))
    try:
        connect_socket.connect(server_address)
        print('\t\tClientOutput:Connected to {}'.format(server_address))
    except ConnectionRefusedError:
        assert False, '\t\tClientOutput:No server is found on {}'.format(server_address)

    j_out = {'name': 'client', 'msg': 'hello', 'time': 'msg 1'}
    j_out_str = jt.json_to_string(j_out)

    st.send_msg(
        connection=connect_socket,
        buflen=BUF_LEN,
        data=j_out_str,
        msg_end=END_MSG
    )

    print(
        st.buffer_to_str(
            data=j_out_str,
            prefix='ClientOutput:OUT',
            tabs=2
        )
    )

    j_in_str = st.receive_msg(connect_socket, buflen=BUF_LEN, msg_end=END_MSG)
    if j_in_str:
        print(
            st.buffer_to_str(
                data=j_in_str,
                prefix='ClientOutput:IN',
                tabs=2
            )
        )

        j_out = {'name': 'client', 'msg': 'hello', 'time': 'msg 3'}
        j_out_str = jt.json_to_string(j_out)

        st.send_msg(
            connection=connect_socket,
            buflen=BUF_LEN,
            data=j_out_str,
            msg_end=END_MSG
        )

        print(
            st.buffer_to_str(
                data=j_out_str,
                prefix='ClientOutput:OUT',
                tabs=2
            )
        )
    else:
        print('\t\tClientOutput:No Data from {}'.format(connect_socket))
    return


def open_server_test():
    mt.get_function_name(ack=True, tabs=0)
    sock = st.open_server(
        server_address=SERVER_ADDRESS,
        ack=True,
        tabs=1
    )

    # OPEN WITH A DIFFERENT THREAD THE CLIENT
    thread = threading.Thread(target=connect_to_server)
    thread.start()

    print('\t\tWaiting for connection {}/{}:'.format(1, 1))
    client_sock, client_address = sock.accept()
    j_in_str = st.receive_msg(client_sock, buflen=BUF_LEN, msg_end=END_MSG)
    if j_in_str:
        print(
            st.buffer_to_str(
                data=j_in_str,
                prefix='IN',
                tabs=2
            )
        )

        j_out = {'name': 'server', 'msg': 'wait', 't': 'msg 2'}
        j_out_str = jt.json_to_string(j_out)
        st.send_msg(
            connection=client_sock,
            buflen=BUF_LEN,
            data=j_out_str,
            msg_end="$#$#"
        )
        print(
            st.buffer_to_str(
                data=j_out_str,
                prefix='OUT',
                tabs=2
            )
        )

        j_in_str = st.receive_msg(client_sock, buflen=BUF_LEN, msg_end=END_MSG)
        if j_in_str:
            print(
                st.buffer_to_str(
                    data=j_in_str,
                    prefix='IN',
                    tabs=2
                )
            )
        else:
            print('\t\tNo Data from {}'.format(client_address))

        # CLIENT is wait for more messages
        # when finished - close client connection

        print('\t\tTerminating connection...')
        client_sock.close()
    else:
        print('\t\tNo Data from {}'.format(client_address))
    return


def get_host_name_test():
    mt.get_function_name(ack=True, tabs=0)
    print('\t{}'.format(st.get_host_name()))
    return


def get_active_ipv4_test():
    mt.get_function_name(ack=True, tabs=0)
    print('\tactive_ipv4: {}'.format(st.get_active_ipv4()))
    return


def buffer_to_str_test():
    mt.get_function_name(ack=True, tabs=0)
    data = 'hi server, how you doing???'  # len(data)==27
    print(st.buffer_to_str(data, prefix='client1', tabs=1, max_chars=27))
    print(st.buffer_to_str(data, prefix='client1', tabs=1, max_chars=26))
    print(st.buffer_to_str(data, prefix='client1', tabs=1, max_chars=15))
    return


def download_file_test():
    mt.get_function_name(ack=True, tabs=0)
    dir_path = mtt.TEMP_FOLDER1
    if not os.path.exists(dir_path):
        mt.create_dir(dir_path)
    dst = '{}/{}.jpg'.format(mtt.TEMP_FOLDER1, mtt.SO_LOGO)
    url = mtt.IMAGES_D[mtt.SO_LOGO]
    ret = st.download_file(url=url, dst_path=dst)
    if not ret:
        print('\tdownload of {} failed'.format(url))
    else:
        print('\t{} exists ? {}'.format(dst, os.path.exists(dst)))
        # st.download_file(url=mtt.IMAGES_D[mtt.SO_LOGO], dst_path=dst)  # check no overwrite - will fail
        mt.delete_dir_with_files(dir_path=mtt.TEMP_FOLDER1)
    return


def get_file_size_by_url_test():
    mt.get_function_name(ack=True, tabs=0)

    urls = [
        mtt.IMAGES_D[mtt.SO_LOGO],
        mtt.IMAGES_D[mtt.KITE],
        mtt.IMAGES_D[mtt.GIRAFFE],
        'https://cdn.sstatic.net/Sites/stackoverflow/img/NoSuchFile.png'
    ]
    for url in urls:
        url_file = url.split('/')[-1]
        s = st.get_file_size_by_url(url=url)
        if s == 'N/A':
            print('\tdownload of {} failed'.format(url_file))
        else:
            print('\tfile {} size is {}'.format(url_file, s))
    return


def get_wifi_pass_test():
    if mt.is_windows():
        mt.get_function_name(ack=True, tabs=0)
        d = st.get_wifi_passwords(ack=True)
        print('\tdict returned: {}'.format(d))
    return


def map_devices_test():
    mt.get_function_name(ack=True, tabs=0)
    # st.map_devices_arp(known_devices=None, ack=True)

    # if you know macs of devices
    known_devices = {
        'F8:1A:67:99:15:E0': 'my super cool gateway at home'
    }
    st.map_devices_arp(
        known_devices=known_devices, ack=True
    )
    return


def get_all_network_info_test():
    mt.get_function_name(ack=True, tabs=0)
    st.get_all_network_info(with_colons=True, ack=True)
    # st.get_all_network_info(with_colons=False, ack=True)
    return


def get_all_ipv4_test():
    mt.get_function_name(ack=True, tabs=0)
    st.get_wifi_ipv4(ack=True)
    st.get_ethernet_ipv4(ack=True)
    return


def get_all_macs_test():
    mt.get_function_name(ack=True, tabs=0)
    st.get_wifi_mac(ack=True)
    st.get_ethernet_mac(ack=True)
    return


def get_active_con_mac_test():
    mt.get_function_name(ack=True, tabs=0)
    st.get_active_con_mac(with_colons=True, ack=True)
    return


def get_mac_address_uuid_test():
    mt.get_function_name(ack=True, tabs=0)
    st.get_mac_address_uuid(ack=True, tabs=1)
    st.get_mac_address_uuid(with_colons=False, ack=True, tabs=1)
    print('\tMac address is {}'.format(st.get_mac_address_uuid()))
    return


def test_all():
    print('{}{}:'.format('-' * 5, mt.get_base_file_and_function_name()))
    open_server_test()
    get_host_name_test()
    get_active_ipv4_test()
    buffer_to_str_test()
    download_file_test()
    get_file_size_by_url_test()
    get_wifi_pass_test()
    map_devices_test()
    get_all_network_info_test()
    get_all_ipv4_test()
    get_all_macs_test()
    get_active_con_mac_test()
    get_mac_address_uuid_test()
    print('{}'.format('-' * 20))
    return
