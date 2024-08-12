import requests
# curl -b "PHPSESSID=PHPSESSIDngpvfv0c7274hbauj7hna8fto7" http://172.30.1.254:8899/captcha.php --output hi.png

def request_captcha_and_save(id):
    data = requests.get("http://172.30.1.254:8899/captcha.php")

    with open(f"./raw/라벨안됨-{id}.png", "wb") as file:
        file.write(data.content)
    
for i in range(1001, 2000):
    request_captcha_and_save(i)