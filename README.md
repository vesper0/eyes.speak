# Eyes Speak

* 카메라를 이용하여 실시간으로 문자를 인식하고, 단어와 문장을 번역한 뒤 음성 피드백을 제공하는 서비스입니다.

## Environment

* 개발 환경입니다.

```
* Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
* 32GB RAM
* Windows 11
* Python 3.9
```

## Prerequite

* 코드를 실행하기 전에 필요한 과정입니다.

```shell
python -m venv .venv
.venv\Scripts\activate

pip install -U pip
pip install wheel
pip install googletrans
pip install pillow==9.0.0
pip install gtts
pip install playsound

pip install openvino-dev

omz_downloader --name horizontal-text-detection-0001
omz_downloader --name text-recognition-resnet-fc
omz_decoder --all
```

## Steps to build

* 프로젝트를 빌드하는 과정입니다.

```shell
.venv\Scripts\activate
main.py
```

## Steps to run

* 프로젝트를 실행하는 과정입니다.

```shell
.venv\Scripts\activate
main.py
```

## Output

![./images/result1.png](./images/result1.png)
![./images/result2.png](./images/result2.png)

## Appendix

* 번역에 사용하는 Papago API는 계정당 하루 1만자 이용 제한이 걸려있습니다. 개인적으로 API 사용 신청을 해서 사용해야 합니다.
* PIL 10.0.0 버전은 textsize를 지원하지 않습니다. 9.0.0 버전이 필요합니다.
