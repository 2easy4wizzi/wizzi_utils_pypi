import socket
import os
import threading
import psutil
from uuid import getnode as _get_mac_uuid
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError
from wizzi_utils.misc import misc_tools as mt


def open_server(server_address: tuple = ('localhost', 10000), ack: bool = True, tabs: int = 1) -> socket:
    """
    :param server_address:
    :param ack:
    :param tabs:
    see open_server_test()
    """
    if ack:
        print('{}Opening server on IP,PORT {}'.format(tabs * '\t', server_address))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(server_address)
    sock.listen(1)
    return sock


def get_host_name() -> str:
    """
    :return: hostname
    try using misc_tools.get_pc_name() instead
    see get_host_name_test()
    """
    hostname = socket.gethostname()
    return hostname


def get_active_ipv4() -> str:
    """
    :return ipv4 address of this computer
    see get_ipv4_test()
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ipv4 = s.getsockname()[0]
    except OSError as e:
        mt.exception_error('No connection: {}'.format(e), real_exception=True)
        ipv4 = 'N/A'
    return ipv4


def send_msg(connection: socket, buflen: int, data: str, msg_end: str) -> None:
    """
    :param connection: the connection of this client/server to the server/client
    :param buflen: needed to split the message if it is longer than 'buflen'
    :param data: string to send
    :param msg_end: special string that notifies when the msg is over(faster than try catch)
        MUST BE A STRING THAT CANT APPEAR ON MESSAGES - e.g. "$#$#"
    see open_server_test()
    """
    data_e = str.encode(data + msg_end)
    data_e_len = len(data_e)
    for i in range(0, data_e_len, buflen):
        chunk_i = data_e[i:i + buflen]
        connection.send(chunk_i)
    return


def receive_msg(connection: socket, buflen: int, msg_end: str) -> str:
    """
    :param connection: the connection of this client/server to the server/client
    :param buflen: needed to receive the message in chunks
    :param msg_end: special string that notifies when the msg is over(faster than try catch)
        MUST BE A STRING THAT CANT APPEAR ON MESSAGES - e.g. "$#$#"
    :return: string of the received data
    see open_server_test()
    """
    data_in = ''
    saw_end_delimiter = False
    while not saw_end_delimiter:
        data_in += connection.recv(buflen).decode('utf-8')
        if not data_in:
            break  # empty transmission
        if data_in.endswith(msg_end):
            data_in = data_in.replace('$#$#', '')
            saw_end_delimiter = True

    return data_in


def buffer_to_str(data: str, prefix: str, tabs: int = 1, max_chars: int = 100) -> str:
    """
    :param data: data as string
    :param prefix: string prefix e.g. 'in', 'out', 'server', 'client'
    :param tabs:
    :param max_chars:
    :return: pretty print of the buffer
    see buffer_to_str_test()
    """
    data_len = len(data)
    data_chars = data_len + 1 if data_len <= max_chars else max_chars

    msg = '{}{}: {} (bytes sent={})'.format(tabs * '\t', prefix, data[:data_chars], data_len)
    if data_len > max_chars:
        msg += ' ... message is too long'
    return msg


def download_file(url: str, dst_path: str = './file', tabs: int = 1) -> bool:
    """
    :param url:
    :param dst_path: where to save the file.
        if dst_path contains non existing folders - it will fail.
            use mt.create_dir(dir) for new dirs
    :param tabs:
    :return: bool 1 for success
    see download_file_test()
    see load_img_from_web() in test_open_cv_tools.py
    """
    filename = url.split('/')[-1]

    def download_progress_hook(count, blockSize, totalSize):
        if totalSize < 0:  # saw this once
            percent = 'N/A'
            size_s = 'size N/A'
        else:
            percent = '{:.1f}%'.format(min(float(count * blockSize) / float(totalSize) * 100.0, 100))
            size_s = mt.convert_size(totalSize)
        print("\r{}Completed: {} from {} - {}".format((tabs + 1) * '\t', percent, size_s, filename), end="")
        return

    ret = False
    if not os.path.exists(dst_path):
        try:
            if mt.is_windows():
                # in windows - add prefix which allows the file name to be longer than the default len (70 chars)
                dst_path = mt.full_path_no_limit(dst_path)
                if len(os.path.basename(dst_path)) > 255:
                    err = 'len(os.path.basename(dst_path))={} exceeds the maximal length which is 255 chars. '
                    err += 'dst_path = {}'
                    mt.exception_error(err.format(len(os.path.basename(dst_path)), os.path.basename(dst_path)),
                                       real_exception=False, tabs=tabs)
                    return False
            msg = 'Downloading from {} to {}'.format(url, dst_path)
            print('{}{}'.format(tabs * '\t', mt.add_color(msg, ops='light_magenta')))
            urlretrieve(url, dst_path, download_progress_hook)
            print()  # release cartridge
            ret = True
        except URLError as e:
            mt.exception_error('{} {}'.format(e, url), real_exception=True, tabs=tabs + 1)
        except FileNotFoundError as e:
            mt.exception_error('{}'.format(e), real_exception=True, tabs=tabs + 1)

    else:
        mt.exception_error(mt.EXISTS.format(dst_path), real_exception=False, tabs=tabs)
    return ret


def get_file_size_by_url(url: str) -> str:
    try:
        obj_info = urlopen(url)
        size_int = int(obj_info.getheader('Content-Length'))
        size_pretty = mt.convert_size(size_bytes=size_int)
    except URLError as e:
        mt.exception_error('with file {} error: {}'.format(url, e), real_exception=True, tabs=1)
        size_pretty = 'N/A'
    return size_pretty


def get_wifi_password(profile_name: str, ack: bool = True) -> str:
    """
    get a wifi saved password on this computer of a specific profile
    :return:
    """
    cmd = 'netsh wlan show profile \"{}\" key=clear'  # wrap the name in-case it has spaces
    netsh_profile_out = mt.run_shell_command_and_get_out(cmd.format(profile_name), ack_cmd=False)
    password = 'Not Found'
    for i, line in enumerate(netsh_profile_out):
        # print('{}){}'.format(i, line))
        if 'Key Content' in line:
            password = line.split(':')[1].strip()
            # print(repr(password))
            break
    if ack:
        print('\t{:<20}| {}'.format(profile_name, password))
    return password


def get_wifi_passwords(ack: bool = True) -> dict:
    """
    get all wifi saved passwords on this computer
    :return:
    """
    profile_to_password_d = {}
    if mt.is_windows():
        # fetch all profiles
        netsh_out = mt.run_shell_command_and_get_out('netsh wlan show profiles', ack_cmd=False)
        profile_to_password_d = {}
        for i, line in enumerate(netsh_out):
            # print('}){}'.format(i, line))
            if 'All User Profile' in line:
                p_name = line.split(':')[1].strip()
                # print(repr(p_name))
                profile_to_password_d[p_name] = None

        # for each profile, fetch password
        if ack:
            print('Profiles and Passwords:')

        for p_name in profile_to_password_d:
            password = get_wifi_password(profile_name=p_name, ack=False)
            profile_to_password_d[p_name] = password
            if ack:
                print('\t{:<20}| {}'.format(p_name, password))
    return profile_to_password_d


def check_if_received(out_lines: list) -> bool:
    found = False
    for i, line in enumerate(out_lines):
        if 'Reply from' in line:
            # print('{}){}'.format(i, line))
            found = False if 'Destination host unreachable' in line else True
            break
    return found


def check_ip(ip: str) -> bool:
    # print('checking ip {}...'.format(ip))
    out_lines = mt.run_shell_command_and_get_out(cmd='ping -n 1 {}'.format(ip), ack_cmd=False, ack_out=False)
    found = check_if_received(out_lines)
    # if found:
    #     print('\tip {} is logged in'.format(ip))
    return found


def check_ip_thread(ip: str, ips_found: list) -> None:
    # print('checking ip {}...'.format(ip))
    out_lines = mt.run_shell_command_and_get_out(cmd='ping -n 1 {}'.format(ip), ack_cmd=False, ack_out=False)
    found = check_if_received(out_lines)
    if found:
        ips_found.append(ip)
        # print('\tip {} is logged in'.format(ip))
    return


def map_brute_force(start_idx: int = 0, end_idx: int = 254, ack: bool = True) -> list:
    """
    :param start_idx:
    :param end_idx:
    :param ack:
    :return: list of ips detected (replied to ping)
    e.g. output
    --------------------------------------------------------------------------------
    local network prefix 192.168.0
    checking from 192.168.0.X where X goes from 0 to 255...
        ip 192.168.0.2 is logged in
        ip 192.168.0.6 is logged in
        ip 192.168.0.8 is logged in
    --------------------------------------------------------------------------------
    Total run time 0:00:04
    """
    ips_found = []
    my_ip = get_active_ipv4()
    if my_ip == 'N/A':
        return ips_found
    pref_ip = my_ip.split('.')[:3]  # take first 3 chars e.g. 192.168.0.8 -> 192.168.0
    pref_ip = '.'.join(pref_ip)
    pref_ip += '.{}'  # add variable format
    if ack:
        print('local network prefix {}'.format(pref_ip))
        print('checking from {} where X goes from {} to {}...'.format(pref_ip.format('X'), start_idx, end_idx))
    pool = list()

    for i in range(start_idx, end_idx):
        address = pref_ip.format(i)
        # check_ip(ip=address)
        x = threading.Thread(target=check_ip_thread, args=(address, ips_found))
        pool.append(x)
        x.start()
    for index, thread in enumerate(pool):
        thread.join()
    ips_found.sort()
    if ack:
        for ip in ips_found:
            print('\tip {} is logged in'.format(ip))
    return ips_found


def map_devices_arp(known_devices: dict = None, ack: bool = True):
    """
    :param known_devices: if you know the mac of a device, give it a nickname and it will be displayed
        e.g.
        known_devices = {
            '70:4D:7B:8A:65:EE': 'Wizzi-Dorms (Ethernet adapter Ethernet)',
            'D4-6E-0E-16-51-41': 'Wizzi-Dorms (Wireless LAN adapter Wi-Fi)',
            'A2:8C:1E:42:57:64': 'Wizzi-Pixel3a',
            '5C:80:B6:30:1A:7E': 'Wizzi-Dell (Wireless LAN adapter Wi-Fi)',
        }
        notice Wizzi-Dorms is my home pc and have 2 mac
        cmd > ipconfig /all
        under Ethernet adapter Ethernet one mac and under Wireless LAN adapter Wi-Fi another
        one for lan and one for wifi
        on linux:
        shell > ifconfig
        wlan0 and eth0 are the ones we need
    :param ack:
    :return:
    TODO check on linux
    Notice that if you get something like:
    File "C:/Users/GiladEiniKbyLake/.conda/envs/bin_env/lib/subprocess.py", line 516,
        in run: Command 'ping -n 1 192.168.0.7' returned non-zero exit status 1.
    it's probably a firewall issue on that device.
    if you wish that device will be discovered and it uses windows firewall,
    on the device, you can add inbound rule to win defender:
        https://activedirectorypro.com/allow-ping-windows-firewall/
    control panel > firewall > advanced > inbound rules
    sort by group and go to group file and printer sharing
    select "File and Printer Sharing (Echo Request - ICMPv4-In)" (should be and entry to private and to any)
    change to enabled and done

    e.g.
    # if you know macs of devices
    known_devices = {
        'F8:1A:67:99:15:E0': 'my super cool gateway at home'
    }
    st.map_devices_arp(
        known_devices=known_devices, ack=True
    )

    it's takes ~10 seconds when a device logs in or out to refresh
    example output with more known devices:
   --------------------------------------------------------------------------------
    arp -a
         IP                MAC           NICKNAME
    192.168.0.2     F8:1A:67:99:15:E0    default gateway Wizzi-Dorms
    192.168.0.5     66:93:D1:80:F9:02
    192.168.0.6     A2:8C:1E:42:57:64    Wizzi-Pixel3a
    192.168.0.7     5C:80:B6:30:1A:7E    Wizzi-Dell (Wireless LAN adapter Wi-Fi)
    192.168.0.10    DC:A6:32:B8:1C:21    rp1-64GB (wlan0)
    192.168.0.8     70:4D:7B:8A:65:EE    Wizzi-Dorms (Ethernet adapter Ethernet)
    --------------------------------------------------------------------------------
    Total run time 0:00:04
    """
    ip_mac = {}
    my_ip = get_active_ipv4()
    if my_ip == 'N/A':
        return ip_mac
    map_brute_force(ack=False)  # refresh arp cache table
    out_lines = mt.run_shell_command_and_get_out(cmd='arp -a', ack_cmd=True, ack_out=False)

    for line in out_lines:
        if 'dynamic' in line:  # break down the output keep only dynamic addresses
            line = ' '.join(line.split())  # many spaces to 1
            line_s = line.split(' ')[:2]  # keep only Internet Address and Physical Address headers
            address, mac = line_s
            ip_mac[address] = mac.replace('-', ':').upper()

    ip_mac[get_active_ipv4()] = get_active_con_mac()  # add this computer
    if ack:
        if ip_mac:
            print(' ' * 4, 'IP', ' ' * 14, 'MAC', ' ' * 9, 'NICKNAME')
        for address, mac in ip_mac.items():
            info = '{:15} {:20}'.format(address, mac)
            if known_devices is not None and mac in known_devices:
                name = known_devices[mac]
                info += ' {}'.format(name)
            print(info)
    return ip_mac


def get_all_network_info(with_colons: bool = True, ack: bool = False) -> dict:
    """
    :param with_colons:
    :param ack:
    :return:
    get all network interfaces (virtual and physical)
    based on https://www.thepythoncode.com/article/get-hardware-system-information-python
    """
    if_addrs = psutil.net_if_addrs()
    info_d = {}
    for interface, interface_addresses in if_addrs.items():
        # print('interface {}:'.format(interface))
        info_d[interface] = {}
        for address in interface_addresses:
            # print('address {}:'.format(interface_addresses))
            k = None
            v = address.address.upper()
            if str(address.family) == 'AddressFamily.AF_INET':
                # from this family we get the ip the interface uses
                k = 'ip'
            elif str(address.family) == 'AddressFamily.AF_INET6':
                # from this family we get the 20 digit mac
                if mt.is_linux():
                    # on linux e.g. address='fe80::8793:7140:756b:1433%wlan0'
                    v = v.split('%')[0].replace('::', ':')
                k = 'mac20' if len(v) == 24 else None  # valid: 20 hexa + 4 semicolons
                if k and not with_colons:
                    v = v.replace(':', '')
            elif str(address.family) in ['AddressFamily.AF_LINK', 'AddressFamily.AF_PACKET']:
                # from this family we get the 12 digit mac (AF_LINK on windows, AF_PACKET on linux)
                k = 'mac12' if len(v) == 17 else None  # valid: 12 hexa + 5 semicolons
                # v is AB-CD-EF-GH-IJ-KL on windows and AB:CD:EF:GH:IJ:KL on linux
                if k:
                    if with_colons:
                        v = v.replace('-', ':')  # handle only windows (linux already in form)
                    else:  # asked to strip delimiter
                        # windows returns xx-yy-... linux returns xx:yy:...
                        v = v.replace('-', '').replace(':', '')  # remove both
            if k:
                info_d[interface][k] = v
    if ack:
        for interface, info in info_d.items():
            print('interface {}:'.format(interface))
            for family_k, address in info.items():
                print('\t{}: {}'.format(family_k, address))

    return info_d


def _get_network_key(interface: str, key: str, with_colons: bool = True) -> str:
    """
    aux function
    :param interface:
    :param key:
    :param with_colons:
    :return:
    """
    mac = None
    info_d = get_all_network_info(with_colons=with_colons, ack=False)
    if interface in info_d:
        if key in info_d[interface]:
            mac = info_d[interface][key]
    return mac


def get_wifi_ipv4(ack: bool = False) -> str:
    """
    :param ack:
    :return: get wifi ip if connected
    """
    interface = None
    if mt.is_windows():
        interface = 'Wi-Fi'
    elif mt.is_linux():
        interface = 'wlan0'
    ip_wifi = _get_network_key(interface=interface, key='ip')
    if ack:
        print('\twifi     ip {}'.format(ip_wifi))
    return ip_wifi


def get_ethernet_ipv4(ack: bool = False) -> str:
    """
    :param ack:
    :return: get ethernet ip if connected
    """
    interface = None
    if mt.is_windows():
        interface = 'Ethernet'
    elif mt.is_linux():
        interface = 'eth0'
    ip_ether = _get_network_key(interface=interface, key='ip')
    if ack:
        print('\tethernet ip {}'.format(ip_ether))
    return ip_ether


def get_wifi_mac(with_colons: bool = True, ack: bool = False) -> str:
    """
    :param with_colons:
    :param ack:
    :return: get wifi mac if connected
    """
    interface = None
    if mt.is_windows():
        interface = 'Wi-Fi'
    elif mt.is_linux():
        interface = 'wlan0'
    mac_wifi = _get_network_key(interface=interface, key='mac12', with_colons=with_colons)
    if ack:
        print('\twifi     mac {}'.format(mac_wifi))
    return mac_wifi


def get_ethernet_mac(with_colons: bool = True, ack: bool = False) -> str:
    """
    :param with_colons:
    :param ack:
    :return: get ethernet mac if connected
    """
    interface = None
    if mt.is_windows():
        interface = 'Ethernet'
    elif mt.is_linux():
        interface = 'eth0'
    mac_ethernet = _get_network_key(interface=interface, key='mac12', with_colons=with_colons)
    if ack:
        print('\tethernet mac {}'.format(mac_ethernet))
    return mac_ethernet


def get_active_con_mac(with_colons: bool = True, ack: bool = False) -> str:
    """
    :param with_colons:
    :param ack:
    :return: mac of wifi/ethernet depends on what is connected and active
    """
    # todo check linux - might be ethe0/wlan0
    active_con = None
    mac12_con = None
    ipv4 = get_active_ipv4()
    if not ipv4 == 'N/A':
        info_d = get_all_network_info(with_colons=with_colons, ack=False)
        for interface, info in info_d.items():
            if 'ip' in info and info['ip'] == ipv4:
                if 'mac12' in info:
                    mac12_con = info['mac12']
                    active_con = interface
        if ack:
            print('\tactive con({}) mac {}'.format(active_con, mac12_con))
    return mac12_con


def get_mac_address_uuid(with_colons: bool = True, ack: bool = False, tabs: int = 1) -> str:
    """
    :return:
    on windows: probably the ethernet mac even if it's disabled
    on linux: active interface mac
    see get_mac_address_test()
    """
    try:
        mac = _get_mac_uuid()
        mac = "{:X}".format(mac)  # turn to Hex
        if with_colons:
            mac = ':'.join(mac[i:i + 2] for i in range(0, 12, 2))
        if ack:
            print('{}* Computer Mac: {}'.format(tabs * '\t', mac))
    except ModuleNotFoundError as e:
        mac = ''
        mt.exception_error(e, real_exception=True)
    return mac
