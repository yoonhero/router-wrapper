from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import requests
from dotenv import load_dotenv
import os
import selenium
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, UnexpectedAlertPresentException
# from io import BytesIO
# from PIL import Image
# from base64 import b64decode

from settings import Device, PortFWDSetting
from inference import Inference

load_dotenv()

username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")

# Just Example
internal_ip = os.getenv("INTERNAL_IP")
edit_infos = [
    PortFWDSetting(external_port_start="22", internal_ip=internal_ip, internal_port_start="2718", description="sss"),
    PortFWDSetting(external_port_start="80", internal_ip=internal_ip, internal_port_start="80", description="http"),
    PortFWDSetting(external_port_start="443", internal_ip=internal_ip, internal_port_start="443", description="httpss"),
    PortFWDSetting(external_port_start="25", internal_ip=internal_ip, internal_port_start="25", description="mail"),
    PortFWDSetting(external_port_start="995", internal_ip=internal_ip, internal_port_start="995", description="mail"),
    PortFWDSetting(external_port_start="587", internal_ip=internal_ip, internal_port_start="587", description="mail"),
    PortFWDSetting(external_port_start="110", internal_ip=internal_ip, internal_port_start="110", description="mail"),
]

to_edit_port_info = [info.external_port_start for info in edit_infos]

cur_device = Device["KT_GIGA_WIFI"]
host_ip = cur_device.host_ip
portfwd_path = cur_device.portfwd_path
login_element_info = cur_device.login_element_info
is_captcha_exist = cur_device.captcha
captcha_url = cur_device.captcha_img_request_url

# Please Edit this threshold or re-define the captcha fidelity metric in inference.py
captcha_min_fidelity = 0.1

basic_puzzle_dir = "./puzzle.png"

driver = webdriver.Chrome()

inference = Inference("model.pt")

def request_captcha_and_save():
    browser_cookies = driver.get_cookies()
    browser_cookies = {cookie['name']: cookie['value'] for cookie in browser_cookies}
    cookies = {'PHPSESSID': browser_cookies["PHPSESSID"]}    
    data = requests.get(f"{host_ip}/{captcha_url}", cookies=cookies)
    with open(basic_puzzle_dir, "wb") as file:
        file.write(data.content)
        # file.write(driver.find_element(By.ID, "captcha").screenshot_as_png)
        time.sleep(1)

def solve_captcha():
    while True:
        request_captcha_and_save()
        # img = Image.open(BytesIO(b64decode(driver.find_element(By.ID, "captcha").screenshot_as_base64)))
        pred_text, fidelity = inference.pred(basic_puzzle_dir)
        print(f"Pred: {pred_text} | Fidelity: {fidelity}")
        if fidelity > captcha_min_fidelity:
            return pred_text
        time.sleep(1)
        # driver.find_element(By.ID, "captcharefresh").click()

def try_login():
    username_input = driver.find_element(By.XPATH, '//*[@id="form1"]/div/div[2]/div/div/table/tbody/tr[2]/td[3]/input')
    username_input.send_keys(username)
    password_input = driver.find_element(By.XPATH, '//*[@id="form1"]/div/div[2]/div/div/table/tbody/tr[3]/td[3]/input')
    password_input.send_keys(password)

    if is_captcha_exist:
        captcha_input = driver.find_element(By.XPATH, '//*[@id="form1"]/div/div[2]/div/div/table/tbody/tr[5]/td[2]/input')
        # captcha_url = 
        captcha_solution = solve_captcha()
        captcha_input.send_keys(captcha_solution)

    login_button = driver.find_element(By.ID, 'login')
    login_button.click()

def check_is_login():
    cookie = driver.get_cookie("USESSION")
    return not cookie == None

while True:
    driver.get(host_ip)
    time.sleep(1)

    try:
        try_login()

        if check_is_login():
            break

    except selenium.common.exceptions.UnexpectedAlertPresentException:
        time.sleep(1)
        WebDriverWait(driver, 2).until(EC.alert_is_present())
        alert = Alert(driver)
        alert.accept()

    # # Handling alert event wisely
    # driver.execute_script(f"window.open('{host_ip}', '_blank');")
    # driver.switch_to.window(driver.window_handles[0])
    # driver.close()
    # driver.switch_to.window(driver.window_handles[1])

time.sleep(3)

def get_existing_portfwd_settings():
    driver.get(f'{host_ip}/{portfwd_path}')

    trs = driver.find_element(By.XPATH, '//*[@id="form1"]/table/tbody/tr[10]/td/table/tbody').find_elements(By.XPATH, "tr")
    trs = trs[1:]
    ports = []
    checkboxes = []

    # //*[@id="form1"]/table/tbody/tr[10]/td/table/tbody/tr[2]/td[4]
    for tr in trs:
        cb = tr.find_element(By.TAG_NAME, 'input')
        t = tr.find_elements(By.TAG_NAME, 'td')[3].text
        port = t.split("-")[0]
        ports.append(port)
        # checkboxes.append(cb)
        
    # return ports, checkboxes
    return ports

def add_portfwd_setting(setting: PortFWDSetting):
    driver.get(f'{host_ip}/{portfwd_path}') 

    source_ip_input = driver.find_element(By.XPATH, '//*[@id="form1"]/table/tbody/tr[2]/td[2]/input')
    source_ip_input.clear()

    source_port_start_input = driver.find_element(By.XPATH, '//*[@id="form1"]/table/tbody/tr[3]/td[2]/input[1]')
    source_port_end_input = driver.find_element(By.XPATH, '//*[@id="form1"]/table/tbody/tr[3]/td[2]/input[2]')
    source_port_start_input.clear()
    source_port_end_input.clear()

    external_port_start_input = driver.find_element(By.XPATH, '//*[@id="form1"]/table/tbody/tr[4]/td[2]/input[1]')
    external_port_end_input = driver.find_element(By.XPATH, '//*[@id="form1"]/table/tbody/tr[4]/td[2]/input[2]')
    external_port_start_input.clear()
    external_port_end_input.clear()

    internal_ip_input = driver.find_element(By.XPATH, '//*[@id="form1"]/table/tbody/tr[5]/td[2]/input[1]')
    internal_ip_input.clear()

    internal_port_start_input = driver.find_element(By.XPATH, '//*[@id="form1"]/table/tbody/tr[6]/td[2]/input[1]')
    internal_port_end_input = driver.find_element(By.XPATH, '//*[@id="form1"]/table/tbody/tr[6]/td[2]/input[2]')
    internal_port_start_input.clear()
    internal_port_end_input.clear()

    description_input = driver.find_element(By.XPATH, '//*[@id="form1"]/table/tbody/tr[8]/td[2]/input')
    description_input.clear()

    source_ip_input.send_keys(setting.source_ip)
    source_port_start_input.send_keys(setting.source_port_start)
    source_port_end_input.send_keys(setting.source_port_end)
    external_port_start_input.send_keys(setting.external_port_start)
    external_port_end_input.send_keys(setting.external_port_end)
    internal_ip_input.send_keys(setting.internal_ip)
    internal_port_start_input.send_keys(setting.internal_port_start)
    internal_port_end_input.send_keys(setting.internal_port_end)
    description_input.send_keys(setting.description)

    time.sleep(0.5)

    save_button = driver.find_element(By.ID, 'portforward_add')
    save_button.click()

ports = get_existing_portfwd_settings()
# for existed_port, checkbox in zip(ports, checkboxes):
#     if existed_port in to_edit_port_info:

#         if not checkbox.is_selected():
#             checkbox.click()
changed = 0
for i, existed_port in enumerate(ports):
    if existed_port in to_edit_port_info:
        changed+=1
        try:
            checkbox = driver.find_element(By.XPATH, f'//*[@id="form1"]/table/tbody/tr[10]/td/table/tbody/tr[{i+2}]').find_element(By.TAG_NAME, "input")
            if not checkbox.is_selected(): checkbox.click()
        except selenium.common.exceptions.NoSuchElementException:
            print("Error??")
            pass

if changed > 0:
    delete_button = driver.find_element(By.ID, 'portforward_del')
    delete_button.click()
    time.sleep(5)

for edit_info in edit_infos:
    add_portfwd_setting(edit_info)
    time.sleep(4) 

driver.quit()