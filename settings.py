from pydantic import BaseModel

class PortFWDSetting(BaseModel):
    source_ip: str = ""
    source_port_start: str = ""
    source_port_end: str = ""
    external_port_start: str = ""
    external_port_end: str = ""
    internal_ip: str = ""
    internal_port_start: str = ""
    internal_port_end: str = ""
    protocol: str = "TCP"
    description: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.source_port_end == "": self.source_port_end = self.source_port_start
        if self.external_port_end == "": self.external_port_end = self.external_port_start
        if self.internal_port_end == "": self.internal_port_end = self.internal_port_start

class LoginElementInfo(BaseModel):
    username_field_id: str
    password_field_id: str
    captcha_field_id: str=""
    captcha_img_request_url: str=""

class DeviceSetting(BaseModel):
    name: str = True
    captcha: bool = True
    captcha_img_request_url: str = "captcha.php"
    login_element_info: LoginElementInfo = LoginElementInfo(username_field_id="loginid", password_field_id="loginpw", captcha_field_id="answer")
    portfwd_path: str
    host_ip: str

Device = {
    "KT_GIGA_WIFI": DeviceSetting(name="KT_GIGA_WIFI", captcha=True, portfwd_path="nat/portfwd", host_ip="http://172.30.1.254:8899")
}